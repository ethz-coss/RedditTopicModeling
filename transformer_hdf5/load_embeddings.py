import reddit_projection
import time
import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import zstandard
from typing import List
import os
import json
import logging
from zst_io import read_lines_zst, write_lines_zst
import h5py
import numpy as np
import duck_try
from sentence_transformers import SentenceTransformer


logger = logging.getLogger('subreddit_extraction')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

#con = duckdb.connect("/cluster/work/coss/anmusso/victoria/loaded_data/loaded_comments.db")


def extract_comments(subreddits: List[str], input_file_name: str, reader_window_size: int = 2 ** 31,
                     reader_chunk_size: int = 2 ** 27):
    lines_in_subreddits = []
    subreddits_ = set(subreddits)

    con.sql(
        "CREATE TABLE comments_20_04 (score INT, subreddit VARCHAR, author VARCHAR, created_utc INTEGER, body VARCHAR "
        ", id VARCHAR, parent_id VARCHAR )")

    bad_lines, file_lines, file_bytes_processed, subreddits_lines = 0, 0, 0, 0
    file_size = os.stat(input_file_name).st_size

    # Loop through every line in the file
    for line, file_bytes_processed in read_lines_zst(file_name=input_file_name, reader_window_size=reader_window_size,
                                                     reader_chunk_size=reader_chunk_size):
        try:
            line_json = json.loads(line)
            #do basic preselection of lines
            if (line_json['subreddit'] in subreddits_) and (len(line_json['body']) > 50) and (line_json['score'] > 2):
                desired_attributes = {'score', 'subreddit', 'author', 'created_utc', 'body', 'id',
                                      'parent_id'}  # Specify the attributes you want to keep
                filtered_line = {key: value for key, value in line_json.items() if key in desired_attributes}
                lines_in_subreddits.append(filtered_line)
                subreddits_lines += 1
        except (KeyError, json.JSONDecodeError) as err:
            bad_lines += 1
        file_lines += 1

        # Log progress
        if file_lines % 200_000 == 0:
            logger.info(
                f": {subreddits_lines:,} {file_lines:,} : {bad_lines:,} : {file_bytes_processed:,}:{(file_bytes_processed / file_size) * 100:.0f}%")

        # Write the lines to the db file
        if len(lines_in_subreddits) > 10_000:  #for now only opening json file once is fastest..
            with open('/cluster/work/coss/anmusso/victoria/temp.json', 'w') as f:
                json.dump(lines_in_subreddits, f)
            con.execute("COPY comments_20_04 FROM  '/cluster/work/coss/anmusso/victoria/temp.json' (FORMAT JSON, AUTO_DETECT true) ;")
            print("added to collection", len(lines_in_subreddits))
            lines_in_subreddits = []

    #last bit of lines
    with open('/cluster/work/coss/anmusso/victoria/temp.json', 'w') as f:
        json.dump(lines_in_subreddits, f)
    con.execute("COPY comments_20_04 FROM  '/cluster/work/coss/anmusso/victoria/temp.json' (FORMAT JSON, AUTO_DETECT true) ;")
    print('total added lines', con.sql("SELECT COUNT(*) FROM comments_20_04"))

    logger.info(f"Complete : {file_lines:,} : {bad_lines:,} : {subreddits_lines:,}")



def extract_submissions(subreddits: List[str], input_file_name: str, reader_window_size: int = 2 ** 31,
                        reader_chunk_size: int = 2 ** 27):
    lines_in_subreddits = []
    subreddits_ = set(subreddits)

    bad_lines, file_lines, file_bytes_processed, subreddits_lines = 0, 0, 0, 0
    file_size = os.stat(input_file_name).st_size

    con.sql(
        "CREATE TABLE submission_13_05 (subreddit VARCHAR, score INT, author VARCHAR, "
        "created_utc INTEGER, title VARCHAR, id VARCHAR, num_comments INTEGER, downs INTEGER )")

    # Loop through every line in the file
    for line, file_bytes_processed in read_lines_zst(file_name=input_file_name, reader_window_size=reader_window_size,
                                                     reader_chunk_size=reader_chunk_size):
        try:
            line_json = json.loads(line)
            #do basic preselection of lines
            if (line_json['subreddit'] in subreddits_) and (len(line_json['title']) > 30) and (line_json['score'] > 20) \
                    and (line_json['num_comments'] > 10) and (line_json['media'] is None):
                desired_attributes = {'subreddit', 'score', 'author', 'created_utc', 'title', 'id', 'num_comments',
                                      'downs'}  # Specify the attributes you want to keep
                filtered_line = {key: value for key, value in line_json.items() if key in desired_attributes}
                lines_in_subreddits.append(filtered_line)
                subreddits_lines += 1
        except (KeyError, json.JSONDecodeError) as err:
            bad_lines += 1
        file_lines += 1

        # Log progress
        if file_lines % 200_000 == 0:
            logger.info(
                f": {subreddits_lines:,} {file_lines:,} : {bad_lines:,} : {file_bytes_processed:,}:{(file_bytes_processed / file_size) * 100:.0f}%")

        # Write the lines to the db file
        if len(lines_in_subreddits) > 200_000:  #for now only opening json file once is fastest..
            with open('./data/temp_10_12.json', 'w') as f:
                json.dump(lines_in_subreddits, f)
            con.execute("COPY submission_13_05 FROM  './data/temp_10_12.json' (FORMAT JSON, AUTO_DETECT true) ;")
            lines_in_subreddits = []

    #last bit of lines
    with open('./data/temp_10_12.json', 'w') as f:
        json.dump(lines_in_subreddits, f)
    con.execute("COPY submission_13_05 FROM  './data/temp_10_12.json' (FORMAT JSON, AUTO_DETECT true) ;")
    print('added lines', con.sql("SELECT COUNT(*) FROM submission_13_05"))

    logger.info(f"Complete : {file_lines:,} : {bad_lines:,} : {subreddits_lines:,}")


def new_load_embeddings_t(source: str, out_file:str):

    #get lines from duck db
    data = duck_try.get_comments(source)
    lines = data['title'].tolist()
    subreddits = data['subreddit'].tolist()
    ids = data['id'].tolist()

    #model = SentenceTransformer("./model/all-MiniLM-L6-v2")
    model = SentenceTransformer("/cluster/work/coss/anmusso/victoria/model/all-MiniLM-L6-v2")
    #print(lines[0:10])
    chunck_size = 10000

    hf = h5py.File('{}{}'.format('/cluster/work/coss/anmusso/victoria/embeddings/',out_file), 'w')
    
    for j in range(0,len(lines),chunck_size):
        embeddings = model.encode(lines[j:min(len(lines),j+chunck_size)])
        embeddings_t = np.array(embeddings, dtype=np.float64)
        for i in range (j,min(len(lines),j+chunck_size)):
            # Create dataset for vector
            hf.create_dataset(f'vector_{i}', data=embeddings_t[i-j])
            # Attach metadata as attribute to the dataset
            hf[f'vector_{i}'].attrs['title'] = lines[i]
            hf[f'vector_{i}'].attrs['subreddit'] = subreddits[i]
            hf[f'vector_{i}'].attrs['id'] = ids[i]
        print("added: ", min(len(lines),j+chunck_size))


    print('done saving embeddings')
    hf.close()



def new_load_embeddings_t2(source: str, out_file:str):

    #get lines from duck db
    data = duck_try.get_comments(source) # change to duck_try.get_submissions() for submissions
    lines = data['title'].tolist()
    subreddits = data['subreddit'].tolist()
    ids = data['id'].tolist()

    #model = SentenceTransformer("./model/all-MiniLM-L6-v2")
    model = SentenceTransformer("/cluster/work/coss/anmusso/victoria/model/all-MiniLM-L6-v2")

    
    embeddings = model.encode(lines)
    embeddings_t = np.array(embeddings, dtype=np.float64)

    #print("embeddings: ", type(embeddings), embeddings[0])
    print("done getting embeddings ")

    hf = h5py.File('{}{}'.format('/cluster/work/coss/anmusso/victoria/embeddings/',out_file), 'w')

    for i, (vector, line, sub, id) in enumerate(zip(embeddings_t, lines, subreddits, ids)):
        # Create dataset for vector
        hf.create_dataset(f'vector_{i}', data=vector)
        # Attach metadata as attribute to the dataset
        hf[f'vector_{i}'].attrs['title'] = line
        hf[f'vector_{i}'].attrs['subreddit'] = sub
        hf[f'vector_{i}'].attrs['id'] = id

    print('done saving embeddings')
    hf.close()


def embeddings_from_file_id(good_ids):
    vectors = []
    titles = []
    subreddits = []
    ids = []

    with h5py.File('vectors_all.h5', 'r') as hf:
        # Read vectors and metadata from the file
        for key in hf.keys():
            if (hf[key].attrs['id'] in good_ids):
                vectors.append(hf[key][:])
                titles.append(hf[key].attrs['title'])
                subreddits.append(hf[key].attrs['subreddit'])
                ids.append(hf[key].attrs['id'])

        hf.close()
    meta = [titles, subreddits, ids]
    M_embedd = np.matrix(vectors)

    print('Membedd: ', M_embedd.shape)

    print("getting vectors: ", len(meta[0]))

    return M_embedd, meta


def embeddings_from_file(source:str):
    vectors = []
    titles = []
    subreddits = []
    ids = []

    with h5py.File('{}{}'.format('/cluster/work/coss/anmusso/victoria/embeddings/',source), 'r') as hf:
        # Read vectors and metadata from the file
        for key in hf.keys():
            vectors.append(hf[key][:])
            titles.append(hf[key].attrs['title'])
            subreddits.append(hf[key].attrs['subreddit'])
            ids.append(hf[key].attrs['id'])

        hf.close()
    meta = [titles, subreddits, ids]
    M_embedd = np.matrix(vectors)

    print('Membedd: ', M_embedd.shape)

    print("getting vectors: ", len(meta[0]))

    return M_embedd, meta


if __name__ == '__main__':

    #con.sql("DROP TABLE IF EXISTS comments_20_05")
    base_path = '/cluster/work/coss/anmusso/reddit'
    input_file_path = f'{base_path}/comments/RC_2020-04.zst'

    subr = ['Republican', 'democrats','healthcare', 'Feminism', 'nra', 'education', 'climatechange',
            'politics','progressive', 'The_Donald','TrueChristian','Trucks','teenagers','AskMenOver30',
            'backpacking','news','BlackLivesMatter','racism','news','usa','DefundPoliceNYC']
    
    
    start = time.time()
    #extract_comments(subreddits=subr, input_file_name=input_file_path)
    #extract_submissions(subreddits=subr, input_file_name=input_file_path)

    source = 'comments_20_05'
    output = 'vectors_C_20_05.h5'
    new_load_embeddings_t(source, output)
    print("done embedding comments_20_05 to file vectors_C_20_05.h5")
    print(f'Time: {time.time() - start}')