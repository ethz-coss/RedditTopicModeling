#import cuml
import time
import duckdb
import numpy as np
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer

from steps import db_queries, embeddings, plotting, load_to_db, topic_finding as tf, UMAP_embeddings as um
import config

duck_database = duckdb.connect(config.DATA_BASE_PATH)


def extract_filter_load():
    start = time.time()

    tablename = 'comments'
    for month in config.MONTHS_COMMENTS:
        input_file = f'{config.INPUT_FILE_BASE_PATH}/comments/RC_2020-{month}.zst'
        load_to_db.extract_comments(input_file_name=input_file, table_name=tablename, sql_db=duck_database)

    tablename = 'submissions'
    for month in config.MONTHS_SUBMISSIONS:
        input_file = f'{config.INPUT_FILE_BASE_PATH}/submissions/RS_2020-{month}.zst'
        load_to_db.extract_submissions(input_file_name=input_file, table_name=tablename, sql_db=duck_database)
        print('done month: ', month, duck_database.sql(f"SELECT COUNT(*) From {tablename}"))

    print('time for loading:', time.time() - start)


def compute_embeddings():
    model = SentenceTransformer(config.MODEL_PATH)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num = 0
    start = time.time()
    for C_or_S in config.EMBEDD_SET:
        if C_or_S == 'C':
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
    for C_or_S in config.ANALYSE_SET:
        if C_or_S == 'S':
            table_name = 'submissions'
        else:
            table_name = 'comments'

        file_path = config.EMBEDDINGS_FILE
        if config.FILTER_UMAP:
            subr_nums = db_queries.get_filtered_submissions(table_name=table_name, sql_db=duck_database)
        else:
            subr_nums = db_queries.get_all_numbers(table_name=table_name, sql_db=duck_database)
        print('got ids')
        M_embedd = embeddings.embeddings_from_file_id(subr_nums, file_path)
        print('done retrieving embeddings', M_embedd.shape)
        M_embedd = M_embedd.astype(np.int8)
        coordinates = um.UMAP_to_df_gpu(M_embedd, n_neighbors=config.UMAP_N_Neighbors,
                                        n_components=config.UMAP_COMPONENTS, min_dist=config.UMAP_MINDIST)
        del M_embedd

        coordinates.columns = [str(i) for i in range(config.UMAP_COMPONENTS)]
        coordinates["num"] = subr_nums
        coordinates.to_parquet(f'{config.EMBEDDINGS_BASE_PATH}/{table_name}_{config.PCA_COMPONENTS}_coordinates.parquet', engine='pyarrow', compression='snappy')
        print('umapped with no problems in ', time.time()-start)


def hdbscan_clustering():
    for C_or_S in config.ANALYSE_SET:
        if C_or_S == 'S':
            info_table = 'submissions_info'
            old_table = 'submissions'
        else:
            info_table = 'comments_info'
            old_table = 'comments'

        #load coordinates from file
        coordinates = pd.read_parquet(f'{config.EMBEDDINGS_BASE_PATH}/{old_table}_{config.PCA_COMPONENTS}_coordinates.parquet', engine='pyarrow')

        scanner = cuml.cluster.hdbscan.HDBSCAN(min_cluster_size=config.HDBS_MIN_CLUSTERSIZE)
        clusters = scanner.fit_predict(coordinates.iloc[:, :-1])

        #save to table

        coordinates["cluster"] = clusters
        duck_database.sql(f"DROP TABLE IF EXISTS umap_coordinates")
        duck_database.sql("CREATE TABLE umap_coordinates AS SELECT * FROM coordinates")

        #join table
        duck_database.sql(f"DROP TABLE IF EXISTS {info_table}")
        duck_database.sql(
            f"CREATE TABLE {info_table} AS SELECT * FROM {old_table} AS sub JOIN umap_coordinates ON (umap_coordinates.num = sub.num); ")
        print('done')


def tfidf_topterms_compute():
    if config.ANALYSE_SET == 'S':
        info_table = 'submissions_info'
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
    if config.ANALYSE_SET == 'S':
        table = 'submissions_info'
    else:
        table = 'comments_info'

    blm_rank = tf.get_term_clusters(config.TERMS, table, duck_database)

    cl_num = blm_rank['cluster'][0]
    if cl_num == -1:
        cl_num = blm_rank['cluster'][1]

    print(blm_rank, 'top cluster: ', cl_num)
    print('size of top cluster: ', sizes[cl_num + 1])
    print('top words of top cluster: ', top_words[cl_num + 1])
    print('top words of second cluster: ', top_words[blm_rank['cluster'][2]] )
    print('top subreddits of top cluster: ', tf.get_clusters_subreddit(cl_num, table, duck_database))

    print('top clusters of BLM subreddit: ', tf.get_subreddit_clusters('BlackLivesMatter', table, duck_database))


if __name__ == '__main__':
    compute_umap()
