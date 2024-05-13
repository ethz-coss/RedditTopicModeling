import plotly.express as px
import plotly.graph_objects as go
import time


#used during embedding, default returns up to 100Mil
def get_titles(table_name, sql_con, start=0, batchsize=100000000):
    df = sql_con.sql(f'SELECT title from {table_name} WHERE num >= {start} and num < {start+batchsize} ;').fetchdf()
    return df['title'].tolist()


#create a way to batch this?
def get_subreddit_numbers(table_name, sql_db, subreddits):
     subr_list_str = ','.join([f"'{sub}'" for sub in subreddits])
     l = sql_db.sql(f"SELECT num FROM {table_name} WHERE subreddit IN ({subr_list_str}) ").fetchnumpy()
     return l['NUM'].tolist()

def get_attributes(attributes, numbers, table_name, sql_db):
    atr_list_str = ','.join([f"{atr}" for atr in attributes])
    num_list_str = ','.join([f"'{num}'" for num in numbers])
    df = sql_db.sql(f"SELECT {atr_list_str} FROM {table_name} WHERE num IN ({num_list_str}) ORDER BY num ").fetchdf()
    return df



def user_subreddit_relation( table_name, sql_db, subreddits):
    subr_list_str = ','.join([f"'{sub}'" for sub in subreddits])

    start = time.time()
    print("number of authors that posted in more than 1 chosen subreddit")
    print(sql_db.sql("SELECT count( DISTINCT R1.author) "
                  f"FROM {table_name} R1, {table_name} R2 "
                  f"Where R1.subreddit IN ({subr_list_str})"
                  f"AND R2.subreddit IN ({subr_list_str})"
                  "AND R1.author != '[deleted]' "
                  "AND R1.author = R2.author "
                  "AND R2.subreddit != R1.subreddit "))
    print('time for query: ', time.time()-start)

    start = time.time()
    print("number of authors making separate submissions in both subreddits:")
    df = sql_db.sql("SELECT R1.subreddit AS S1, R2.subreddit AS S2, Count( DISTINCT R1.author) AS Authors "
                 f"FROM {table_name} R1, {table_name} R2 "
                 f"Where R1.subreddit IN ({subr_list_str})"
                 f"AND R2.subreddit IN ({subr_list_str})"
                 "AND R1.author != '[deleted]' "
                 "AND R1.author = R2.author "
                 "AND R1.id != R2.id "  #includes count authors that posted in the same subreddit twice
                 "GROUP BY S1, S2 "
                 #"HAVING S1 < S2 "  #also requires setting this to <=, else <
                 "HAVING S1 != S2 "
                 "ORDER BY Authors DESC ").fetchdf()
    print('time for query (and fetching df): ', time.time()-start)
    
    print(df)
    try:
        pivot_df = df.pivot(index='S1', columns='S2', values='Authors')
        pivot_df.fillna(0, inplace=True)
        print(pivot_df)

        fig = px.imshow(pivot_df,
                        labels=dict(x="subreddit", y="subreddit", color="Authors in both"),
                        x=pivot_df.columns, y=pivot_df.index,
                        )
        fig.update_xaxes(side="top")
        fig.show()
    except: print("couldnt show diagramm")
    


def subreddit_scores(sql_db):
    df = sql_db.sql("WITH Scores AS ( "
                 "SELECT reddit_data.subreddit, COUNT() AS large_score FROM reddit_data WHERE score > 100 GROUP BY subreddit),"
                 "Comments AS ( "
                 "SELECT reddit_data.subreddit, COUNT(*) AS many_comments FROM reddit_data WHERE num_comments > 10 GROUP BY subreddit),"
                 "Topic AS ("
                 "SELECT reddit_data.subreddit, COUNT(*) AS BLM_submission FROM reddit_data "
                 "WHERE reddit_data.title LIKE '%BLM%' "
                 "OR reddit_data.title LIKE '%blm%' "
                 "OR reddit_data.title LIKE '%racism%' "
                 "GROUP BY reddit_data.subreddit)"
                 "SELECT RD.subreddit, Count(*) AS total_submissions, SUM(RD.num_comments) AS total_comments, "
                 "ANY_VALUE(Scores.large_score) AS large_upvote_submissions, "
                 "ANY_VALUE(Comments.many_comments) AS many_comment_submissions, "
                 "ANY_VALUE(Topic.BLM_submission) AS topic_submissions, "
                 "AVG(RD.upvote_ratio) AS avg_upvote_ratio "
                 "FROM reddit_data AS RD "
                 "LEFT JOIN Scores ON RD.subreddit = Scores.subreddit "
                 "LEFT JOIN Comments ON RD.subreddit = Comments.subreddit "
                 "LEFT JOIN Topic ON RD.subreddit = Topic.subreddit "
                 "GROUP BY RD.subreddit").fetchdf()
    print(df.head())
    df.fillna(0, inplace=True)
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns),
                    fill_color='lightblue',
                    align='left'),
        cells=dict(values=df.transpose().values,
                   fill_color='white',
                   align='left'))
    ])
    fig.show()


