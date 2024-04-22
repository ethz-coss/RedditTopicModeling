import duckdb
import json
import os
from load_reddit_data.zst_io import read_lines_zst
import pandas as pd

con = duckdb.connect("file.db")


def read_data_file(input_file_name: str) -> pd.DataFrame:
    # Create a list to store the interesting lines
    interesting_lines = []

    # Mostly for logging purposes
    bad_lines, file_lines, file_bytes_processed = 0, 0, 0
    file_size = os.stat(input_file_name).st_size

    # Loop through every line in the file
    for line, file_bytes_processed in read_lines_zst(file_name=input_file_name):
        try:
            # Load the line as JSON and check if the subreddit is in the list
            line_json = json.loads(line)
            # print(line_json, len(line_json))
            # Do whatever you want with the line (here I print the title of the submission with some other metadata if it in the nra subreddit and then add it to the interesting_lines list)
            interesting_lines.append(line_json)

        # If there are some errors we just skip the line and add it to the bad_lines count
        except (KeyError, json.JSONDecodeError) as err:
            bad_lines += 1

        file_lines += 1

        # I create a pandas dataframe from the interesting lines list
    df_lines = pd.DataFrame(interesting_lines)
    return df_lines


def load_into_duck():
    input_file = './data/RS_2020-06_filtered.zst'
    lines = read_data_file(input_file)
    # we have to drop a few attributes for which sql is not able to determine a Type
    clean = lines.drop(['author_flair_background_color', 'link_flair_background_color', 'author_flair_css_class',
                        'author_flair_template_id', 'author_flair_text', 'gildings', 'all_awardings', 'category',
                        'content_categories', 'discussion_type', 'distinguished', 'removed_by_category',
                        'suggested_sort', 'thumbnail_height', 'thumbnail_width', 'top_awarded_type',
                        'media_embed', 'secure_media', 'secure_media_embed',
                        'treatment_tags', 'url_overridden_by_dest', 'post_hint', 'preview', 'poll_data',
                        'crosspost_parent', 'crosspost_parent_list', 'author_cakeday',
                        'media_metadata', 'event_end', 'event_is_live', 'event_start', 'collections'], axis=1
                       )
    print(clean.columns)
    con.sql("DROP TABLE IF EXISTS reddit_data")
    con.execute("SET GLOBAL pandas_analyze_sample=100000")
    con.sql(" CREATE TABLE reddit_data AS SELECT * FROM clean ")


# "SELECT author, created_utc, id, selftext, subreddit_subscribers, subreddit, "
#              "score, num_comments, wls, upvote_ratio, is_original_content, title "

def get_good_ids():
    df = con.sql("SELECT id "
                 "FROM reddit_data "
                 "WHERE num_comments>10 "
                 "AND (subreddit_subscribers/score) < 1000000 "
                 "AND LEN(title)>30 "
                 "AND media IS NULL ;"
                 ).fetchdf()
    return df

def get_all_ids():
     df = con.sql("SELECT id "
                 "FROM reddit_data "
                 ).fetchdf()
     print(len(df))
     return df

def user_subreddit_relation():
    #returns authors and subreddits they posted in
    print(con.sql("SELECT R1.author , R1.subreddit AS S1, R2.subreddit AS S2 "
                  "FROM reddit_data R1, reddit_data R2 "
                  "Where R1.author_fullname != '[deleted]' "
                  "AND R1.author_fullname = R2.author_fullname "
                  "AND R2.subreddit != R1.subreddit "))

    # number of authors that posted in more than 1 chosen subreddit
    print(con.sql("SELECT count( DISTINCT R1.author_fullname) "
                  "FROM reddit_data R1, reddit_data R2 "
                  "Where R1.author_fullname != '[deleted]' "
                  "AND R1.author_fullname = R2.author_fullname "
                  "AND R2.subreddit != R1.subreddit "))


    print(con.sql("SELECT count(*) "
                  "FROM reddit_data R1, reddit_data R2 "
                  "Where R1.author_fullname != '[deleted]' "
                  "AND R1.author_fullname = R2.author_fullname "
                  "AND R2.subreddit != R1.subreddit "))

    print(con.sql("SELECT count(DISTINCT author_fullname) FROM reddit_data WHERE author != '[deleted]' "))

    print(con.sql("SELECT R1.subreddit AS S1, R2.subreddit AS S2, Count( DISTINCT R1.author_fullname) AS Authors "
                  "FROM reddit_data R1 "
                  "LEFT JOIN reddit_data R2 ON R1.author_fullname = R2.author_fullname "
                  "Where R1.author_fullname != '[deleted]' "
                  "AND R1.id != R2.id " #includes count authors that posted in the same subreddit twice
                  "GROUP BY S1, S2 "
                  "HAVING S1 <= S2 " #also requires setting this to <=, else <
                  "ORDER BY Authors DESC "))

if __name__ == '__main__':
    get_all_ids()
    load_into_duck()
    user_subreddit_relation()
