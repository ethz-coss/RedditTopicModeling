import json
import logging
import os
from typing import Any
import numpy as np
import projection
import h5py
from zst_io import read_lines_zst
import duck_try
from sentence_transformers import SentenceTransformer

EMBEDDING_DIM = 384  # for transformers Model

# Create a logger to output progress in a pretty way
logger = logging.getLogger('example_logger')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)


def example_read_data(input_file_name: str) -> list[Any]:
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
            interesting_lines.append(line_json)

        # If there are some errors we just skip the line and add it to the bad_lines count
        except (KeyError, json.JSONDecodeError) as err:
            bad_lines += 1

        file_lines += 1

    logger.info(f": Bad lines: {bad_lines:,} Read file: {(file_bytes_processed / file_size) * 100:.0f}%")

    return interesting_lines


# this chooses which metadata fields to keep
def relevant_data(lines):
    # choose metadata values
    return {key: lines[key] for key in lines.keys()
            & {'author', 'created_utc', 'id', 'selftext', 'subreddit_subscribers', 'subreddit', 'score', 'num_comments',
               'wls', 'upvote_ratio', 'is_original_content', 'title'}}


def new_load_embeddings_t(source: str):
    #get lines from zst file
    #lines = example_read_data(source)
    #metadata = [relevant_data(lines[i]) for i in range(0, len(lines))]  # get data we want
    #lines = [lines[i]['title'] for i in range(0, len(lines))]

    #get lines from duck db
    data = duck_try.get_comments(source)
    lines = data['title'].tolist()
    subreddits = data['subreddit'].tolist()
    ids = data['id'].tolist()

    #model = SentenceTransformer("./model/all-MiniLM-L6-v2")
    model = SentenceTransformer("/cluster/work/coss/anmusso/victoria/model/all-MiniLM-L6-v2")
    #print(lines[0:10])
    embeddings = model.encode(lines)
    embeddings_t = np.array(embeddings, dtype=np.float64)

    #print("embeddings: ", type(embeddings), embeddings[0])
    print("done getting embeddings ")

    hf = h5py.File('/cluster/work/coss/anmusso/victoria/embeddings/vectors_C_20_05.h5', 'w')

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


def embeddings_from_file():
    vectors = []
    titles = []
    subreddits = []

    with h5py.File('/cluster/work/coss/anmusso/victoria/embeddings/vectors_C_20_05.h5', 'r') as hf:
        # Read vectors and metadata from the file
        for key in hf.keys():
            vectors.append(hf[key][:])
            titles.append(hf[key].attrs['title'])
            subreddits.append(hf[key].attrs['subreddit'])

        hf.close()
    meta = [titles, subreddits]
    M_embedd = np.matrix(vectors)

    print('Membedd: ', M_embedd.shape)

    print("getting vectors: ", len(meta[0]))

    return M_embedd, meta


if __name__ == '__main__':
    # Change this to the path of the file you want to read on your computer

    input_file = './data/RS_2020-06_filtered.zst'
    # embedds all comments in the file and saves them in a collection
    new_load_embeddings_t(source=input_file)
    print('Done embedding')

    ref_0 = ['progressive']  # left of axis
    ref_1 = ['Republican']  # right end of axis
    """
    data = ['Republican', 'Democrats', 'healthcare', 'Feminism', 'nra', 'education', 'climatechange', 'politics',
            'random', 'teenagers', 'progressive', 'The_Donald', 'TrueChristian', 'Trucks', 'AskMenOver30',
            'backpacking']
    """
    data1 = ['climatechange', 'Feminism', 'backpacking', 'progressive']

    data2 = ['Trucks', 'nra', 'AskMenOver30', 'TrueChristian']
    data3 = ['healthcare', 'education']

    df = duck_try.get_good_ids()
    ids = list(df['id'])

    # extracts embeddings only for comments in a certain subreddit, and creates axis between ref0 and ref1
    M_embedd, meta0 = embeddings_from_file_id(collection_name=collection_name, subreddits=ref_0, ids=ids)
    #M_embedd, meta0 = embeddings_from_collection(collection_name=collection_name, subreddits=ref_0)
    avg_0 = projection.average_embedding_n(M_embedd)

    M_embedd, meta1 = embeddings_from_file_id(collection_name=collection_name, subreddits=ref_1, ids=ids)
    #M_embedd, meta1 = embeddings_from_collection(collection_name=collection_name, subreddits=ref_1)
    avg_1 = projection.average_embedding_n(M_embedd)

    axis = projection.create_axis_n(avg_0, avg_1)  # goes from 0 to 1
    print("axis: ", axis)
    print("avg_0: ", projection.project_embedding(axis, avg_0.transpose()))
    print("avg_1: ", projection.project_embedding(axis, avg_1.transpose()))

    # extract comments from subreddits we want to project and project them
    #M_embedd1, meta_data1 = embeddings_from_collection_id(collection_name=collection_name, subreddits=data1, ids = ids)
    M_embedd1, meta_data1 = embeddings_from_file(collection_name=collection_name, subreddits=data1)
    results = np.matmul(M_embedd1, axis.transpose())
    # I should find a smarter way to do this
    results = [float(x[0]) for x in results]
    #print(results)
    try:
        projection.show_stacked_hist(results, meta_data1, "subreddit", num_bins=30, title='left')
    except:
        projection.show_hist(results, title='left')

    #M_embedd2, meta_data2 = embeddings_from_collection_id(collection_name=collection_name, subreddits=data2, ids=ids)
    M_embedd2, meta_data2 = embeddings_from_file(collection_name=collection_name, subreddits=data2)
    results = np.matmul(M_embedd2, axis.transpose())
    # I should find a smarter way to do this
    results = [float(x[0]) for x in results]

    #print(results)
    try:
        projection.show_stacked_hist(results, meta_data2, "subreddit", num_bins=30, title='right')
    except:
        projection.show_hist(results, title='right')

    #M_embedd3, meta_data3 = embeddings_from_file_id(collection_name=collection_name, subreddits=data3, ids=ids)
    M_embedd3, meta_data3 = embeddings_from_file(collection_name=collection_name, subreddits=data3)
    results = np.matmul(M_embedd3, axis.transpose())
    # I should find a smarter way to do this
    results = [float(x[0]) for x in results]
    #print(results)
    try:
        projection.show_stacked_hist(results, meta_data3, "subreddit", num_bins=30, title='neutral')
    except:
        projection.show_hist(results, title='neutral')
