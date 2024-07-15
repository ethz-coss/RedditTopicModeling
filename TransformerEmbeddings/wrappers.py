#import cuml
import time
import duckdb
import numpy as np
import torch
import pandas as pd
import local
from sentence_transformers import SentenceTransformer

from steps import db_queries, embeddings, hdbscan, plotting, load_to_db, topic_finding as tf, UMAP_embeddings as um, plotting as plot
import config


duck_database = duckdb.connect(config.DATA_BASE_PATH)


def extract_filter_load():
    start = time.time()
    if config.C_or_S == 'C':
        tablename = 'comments'
        for month in config.MONTHS:
            input_file = f'{config.INPUT_FILE_BASE_PATH}/comments/RC_2020-{month}.zst'
            load_to_db.extract_comments(input_file_name=input_file, table_name=tablename, sql_db=duck_database)
    else:
        tablename = 'submissions'
        for month in config.MONTHS:
            input_file = f'{config.INPUT_FILE_BASE_PATH}/submissions/RS_2020-{month}.zst'
            load_to_db.extract_submissions(input_file_name=input_file, table_name=tablename, sql_db=duck_database)

    print('time for loading:', time.time() - start)


def compute_embeddings():
    model = SentenceTransformer(config.MODEL_PATH)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num = 0
    start = time.time()

    if config.C_or_S == 'C':
        tablename = 'comments'
    else:
        tablename = 'submissions'

    embeddings_file = config.EMBEDDINGS_FILE
    num += embeddings.new_load_embeddings(table_name=tablename, sql_db=duck_database, output_file=embeddings_file,
                                          model=model, device=device)

    #logg time with gpu
    with open('time_tracking.txt', 'a') as f:  #loggs time into file for comparison
        f.write(f"size:{num} model:{config.MODEL_PATH}, gpu:{torch.cuda.get_device_name(0)} , time:{time.time() - start} \n")



def compute_umap():
    start = time.time()

    if config.C_or_S == 'S':
        table_name = 'submissions'
    else:
        table_name = 'comments'

    file_path = config.EMBEDDINGS_FILE

    subr_nums = db_queries.get_all_numbers(table_name=table_name, sql_db=duck_database)
    #subr_nums = [subr_nums[i] for i in range(0,2_000_000,2)]
    #subr_nums = subr_nums[300_000:1_300_000]
    coordinates = um.UMAP_embeddings(subr_nums=subr_nums, file_path=file_path)

    coordinates["num"] = subr_nums
    coordinates.to_parquet(config.COORDINATES_FILE, engine='pyarrow', compression='snappy')
    print('umapped with no problems in ', time.time()-start)


def hdbscan_clustering():

    if config.C_or_S == 'S':
        info_table = 'submissions_info_filtered'
        old_table = 'submissions'
    else:
        info_table = 'comments_info'
        old_table = 'comments'

    # load coordinates from file
    coordinates = pd.read_parquet(config.COORDINATES_FILE, engine='pyarrow')
    #adds clusters joined on num to info_table
    hdbscan.hdbscan_coordinates(coordinates=coordinates, old_table=old_table, info_table=info_table, sql_db=duck_database)


def tfidf_topterms_compute():
    if config.C_or_S == 'S':
        info_table = 'submissions_info_filtered'
        old_table = 'submissions'
    else:
        info_table = 'comments_info'
        old_table = 'comments'

    tfidf, terms = tf.get_tfidf(info_table, duck_database)
    top_words = tf.get_top_terms(tfidf, terms)

    sizes = tf.topic_sizes(info_table, duck_database)
    sizes_list = sizes['sizes'].tolist()
    return top_words, sizes_list



def show_info(top_words, sizes):
    if config.C_or_S == 'S':
        table = 'submissions_info_filtered'
    else:
        table = 'comments_info'

    blm_rank = tf.get_term_clusters(config.TERMS, table, duck_database)

    cl_num = blm_rank['cluster'][0]
    if cl_num == -1:
        cl_num = blm_rank['cluster'][1]

    print('total clusters:', len(sizes))
    print(blm_rank, 'top cluster: ', cl_num)
    print('size of top cluster: ', sizes[cl_num + 1])
    print('top words of top cluster: ', top_words[cl_num + 1])

    cl_2 = blm_rank['cluster'][2]
    print('size of 2. cluster: ', sizes[cl_2 + 1])
    print('top words of 2. cluster: ', top_words[cl_2 + 1])

    cl_2 = blm_rank['cluster'][3]
    print('size of 3. cluster: ', sizes[cl_2 + 1])
    print('top words of 3. cluster: ', top_words[cl_2 + 1])

    print('top subreddits of top cluster: ', tf.get_clusters_subreddit(cl_num, table, duck_database))

    print('top clusters of BLM subreddit: ', tf.get_subreddit_clusters('BlackLivesMatter', table, duck_database))
    return cl_num

def do_plots(cl_num):
    duck_database = duckdb.connect(config.DATA_BASE_PATH)

    info_table = 'submissions_info_filtered'

    tfidf, terms = tf.get_tfidf(info_table, duck_database)
    top_words = tf.get_top_terms(tfidf, terms)
    sizes = tf.topic_sizes(info_table, duck_database)
    sizes = sizes['sizes'].tolist()

    candidates = [cl_num]

    for n in candidates:
        print('top cluster: ', n)
        print('size of top cluster: ', sizes[n + 1])
        print('top words of top cluster: ', top_words[n + 1])
        print('top subreddits of top cluster: ', tf.get_clusters_subreddit(n, info_table, duck_database))
        plot.unique_per_day(info_table, duck_database, n)
        plot.top_subreddits(info_table, duck_database, n)
        plot.submissions_per_authors(info_table, duck_database, n)
        plot.check(info_table, duck_database, n)

if __name__ == '__main__':
    extract_filter_load()
    compute_embeddings()
    compute_umap()
    hdbscan_clustering()
    top_words, sizes = tfidf_topterms_compute()
    cl_num = show_info(top_words, sizes)
    do_plots(cl_num)

