import json
import logging
import os
from typing import Any

import projection
from zst_io import read_lines_zst

import example
import numpy as np

# EMBEDDING_DIM = 5120 # for 13 B model
EMBEDDING_DIM = 4096  # for 7B model

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
            & {'subreddit', 'score', 'num_comments', 'wls'}}


def new_load_embeddings(source: str, collection_name: str):
    lines = example_read_data(source)
    collection = example.chroma_client.get_or_create_collection(collection_name)
    meta = [relevant_data(lines[i]) for i in range(0, len(lines))]  # get data we want

    print("lines: ", len(lines))
    for i in range(0, len(lines)):
        # getting embedding and adding vector to collection
        embedding = example.get_embedding(lines[i]['title'])
        collection.add(
            ids=[str(i)],
            embeddings=[embedding.tolist()],
            metadatas=[meta[i]]
        )
        if i % 10 == 0:
            print(collection.count())


def embeddings_from_collection(collection_name: str, subreddits: [str]):
    # get all vectors from collection and save them in matrix
    collection = example.chroma_client.get_or_create_collection(collection_name)
    stored = collection.get(
        ids=[str(i) for i in range(0, collection.count())],
        include=["embeddings", "metadatas"],
        where={"subreddit": {"$in": subreddits}}
    )
    embedds = stored["embeddings"]
    meta = stored["metadatas"]

    print("getting vectors: ", len(meta))

    M_embedd = np.matrix(embedds)  # does this work?

    return M_embedd, meta


if __name__ == '__main__':
    # Change this to the path of the file you want to read on your computer
    input_file = 'C:/Users/victo/PycharmProjects/RedditProject/data/RS_2020-06_filtered.zst'
    collection_name = 'Reddit-Comments'

    # embedds all comments in the file and saves them in a collection
    new_load_embeddings(source=input_file, collection_name=collection_name)
    print('Done embedding')

    ref_0 = 'Democrats'  # left of axis
    ref_1 = 'Republican'  # right end of axis
    data = ['Feminism', 'nra']  # what we are projecting

    # extracts embeddings only for comments in a certain subreddit, and creates axis between ref0 and ref1
    M_embedd, meta0 = embeddings_from_collection(collection_name=collection_name, subreddits=ref_0)
    avg_0 = projection.average_embedding(M_embedd)

    M_embedd, meta1 = embeddings_from_collection(collection_name=collection_name, subreddits=ref_1)
    avg_1 = projection.average_embedding(M_embedd)

    axis = projection.create_axis(avg_0, avg_1)  # goes from 0 to 1
    print("axis: ", axis)

    # extract comments from subreddits we want to project and project them
    M_embedd, meta_data = embeddings_from_collection(collection_name=collection_name, subreddits=data)

    results = np.matmul(M_embedd, axis)

    projection.show_hist(results)
