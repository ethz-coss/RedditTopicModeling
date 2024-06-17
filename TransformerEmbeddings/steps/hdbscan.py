import cuml
import time
import duckdb
import numpy as np
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from steps import db_queries, embeddings, plotting, load_to_db, topic_finding as tf, UMAP_embeddings as um
import config


def hdbscan_coordinates(coordinates, old_table, info_table, sql_db):
    scanner = cuml.cluster.hdbscan.HDBSCAN(min_cluster_size=config.HDBS_MIN_CLUSTERSIZE, min_samples=config.HDBS_MIN_SAMPLES)
    clusters = scanner.fit_predict(coordinates.iloc[:, :-1])  #(last collum is num, not a coordinate)

    # save to table
    coordinates["cluster"] = clusters
    sql_db.sql(f"DROP TABLE IF EXISTS umap_coordinates")
    sql_db.sql("CREATE TABLE umap_coordinates AS SELECT * FROM coordinates")

    # join table
    sql_db.sql(f"DROP TABLE IF EXISTS {info_table}")
    sql_db.sql(
        f"CREATE TABLE {info_table} AS SELECT * FROM {old_table} AS sub JOIN umap_coordinates ON (umap_coordinates.num = sub.num); ")
    print('done')
