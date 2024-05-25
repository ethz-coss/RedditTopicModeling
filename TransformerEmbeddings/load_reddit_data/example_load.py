import json
import logging
import os
from transformer_hdf5.zst_io import read_lines_zst
import pandas as pd

# Create a logger to output progress in a pretty way
logger = logging.getLogger('example_logger')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)


def example_read_data(input_file_name: str) -> pd.DataFrame:
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

            # Do whatever you want with the line (here I print the title of the submission with some other metadata if it in the nra subreddit and then add it to the interesting_lines list)
            if line_json['subreddit'] == 'nra':
                print(line_json)
                #print('Title:', line_json['title'], 'Author:', line_json['author'], 'Subreddit:',
                      #line_json['subreddit'], 'Upvotes:', line_json['score'])

                interesting_lines.append(line_json)

        # If there are some errors we just skip the line and add it to the bad_lines count
        except (KeyError, json.JSONDecodeError) as err:
            bad_lines += 1

        file_lines += 1

        # Every 100_000 we log the progress
        if file_lines % 100_000 == 0:
            logger.info(f": Bad lines: {bad_lines:,} Read file: {(file_bytes_processed / file_size) * 100:.0f}%")

    # I create a pandas dataframe from the interesting lines list
    df_lines = pd.DataFrame(interesting_lines)
    return df_lines


if __name__ == '__main__':
    # Change this to the path of the file you want to read on your computer
    input_file = 'C:/Users/victo/PycharmProjects/RedditProject/data/RS_2020-06_filtered.zst'
    example_read_data(input_file_name=input_file)
