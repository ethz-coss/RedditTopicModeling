import zstandard
from typing import List
import os
import json
import logging
from transformer_hdf5.zst_io import read_lines_zst, write_lines_zst

logger = logging.getLogger('subreddit_extraction')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)


def extract_subreddits(subreddits: List[str], input_file_name: str, output_file_name: str, reader_window_size: int = 2 ** 31, reader_chunk_size: int = 2 ** 27):
    lines_in_subreddits = []
    subreddits_ = set(subreddits)

    writer = zstandard.ZstdCompressor().stream_writer(open(output_file_name, 'wb'))

    bad_lines, file_lines, file_bytes_processed, subreddits_lines = 0, 0, 0, 0
    file_size = os.stat(input_file_name).st_size

    # Loop through every line in the file
    for line, file_bytes_processed in read_lines_zst(file_name=input_file_name, reader_window_size=reader_window_size, reader_chunk_size=reader_chunk_size):
        try:
            line_json = json.loads(line)
            #do basic preselection of lines
            if (line_json['subreddit'] in subreddits_):
                print(line_json.items())
                lines_in_subreddits.append(line)
                subreddits_lines += 1
        except (KeyError, json.JSONDecodeError) as err:
            bad_lines += 1
        file_lines += 1

        # Log progress
        if file_lines % 100_000 == 0:
            logger.info(f": {subreddits_lines:,} {file_lines:,} : {bad_lines:,} : {file_bytes_processed:,}:{(file_bytes_processed / file_size) * 100:.0f}%")

        # Write the lines to the output file
        if len(lines_in_subreddits) > 10_000:
            write_lines_zst(writer=writer, lines=lines_in_subreddits)
            lines_in_subreddits = []

    logger.info(f"Complete : {file_lines:,} : {bad_lines:,} : {subreddits_lines:,}")


def example_subreddit_extraction():
    base_path = './../data'
    input_file_path = f'{base_path}/RC_2010-12.zst'
    output_file_path = f'{base_path}/RC_2010-12_filtered.zst'
    subr = ['Republican',
            'democrats',
            'healthcare',
            'Feminism',
            'nra',
            'education',
            'climatechange',
            'politics',
            'progressive',
            'The_Donald',
            'TrueChristian',
            'Trucks',
            'teenagers',
            'AskMenOver30',
            'backpacking',
            'news',
            'BlackLivesMatter',
            'racism',
            'news',
            'usa',
            'DefundPoliceNYC']
    extract_subreddits(subreddits=subr, input_file_name=input_file_path, output_file_name=output_file_path)


example_subreddit_extraction()