import time
import duckdb
import torch
from sentence_transformers import SentenceTransformer
import cuml
import plotly.express as px

import load_to_db
import embeddings
import db_queries
import embedding_graphs.UMAP_embeddings as um



subr = ['Republican', 'democrats', 'healthcare', 'Feminism', 'nra', 'education', 'climatechange',
        'politics', 'progressive', 'The_Donald', 'TrueChristian', 'Trucks', 'teenagers', 'AskMenOver30',
        'backpacking', 'news', 'BlackLivesMatter', 'racism', 'news', 'usa', 'DefundPoliceNYC']

# this is the file to which the sql database will be saved
db_file_c = duckdb.connect("/cluster/work/coss/anmusso/victoria/loaded_data/loaded_comments.db")
db_file_s = duckdb.connect("/cluster/work/coss/anmusso/victoria/loaded_data/loaded_submissions.db")

#adjust file basepath
def load(C_or_S, month, year):
    #load comments into database

    start = time.time()
    if C_or_S == 'C':
        sql_db = db_file_c
        tablename = 'comments'
        input_file = f'/cluster/work/coss/anmusso/reddit/comments/RC_{year}-{month}.zst'
        load_to_db.extract_comments(input_file_name=input_file, table_name=tablename, sql_db=sql_db)
    else:
        sql_db = db_file_s
        tablename = 'submissions'
        input_file = f'/cluster/work/coss/anmusso/reddit/submissions/RS_{year}-{month}.zst'
        load_to_db.extract_submissions(input_file_name=input_file, table_name=tablename, sql_db=sql_db)
    
    
    print('time for loading:', time.time() - start)


#adjust file basepaths
def embedd(C_or_S, model_name = 'all-MiniLM-L6-v2'):
    model = SentenceTransformer(f"/cluster/work/coss/anmusso/victoria/model/{model_name}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if C_or_S == 'C':
        sql_db = db_file_c
        tablename = 'comments'
    else:
        sql_db = db_file_s
        tablename = 'submissions'

    embeddings_file = f'./embeddings_{tablename}_{model_name}.h5py'
    start = time.time()
    num = embeddings.new_load_embeddings(table_name=tablename, sql_db=sql_db, output_file=embeddings_file, model=model, device = device)
    print('time for embedding:', time.time() - start)
    return num

def fetch_embeddings(num_list, table_name, model_name):
    file_path = f"/cluster/work/coss/anmusso/victoria/embeddings/embeddings_{table_name}_{model_name}.h5py"
    return embeddings.embeddings_from_file_id(num_list, file_path)


def try_embedding_models():
    t = 'S' #submissions = S, comments = C
    
    model_name = 'multi-qa-mpnet-base-dot-v1'
    start = time.time()
    num = embedd(t, model_name=model_name) #returns number of embeddings calculated
    with open('time_tracking_2.txt', 'a') as f:
        f.write(f"set: {t}, size:{num} model:{model_name}, gpu:{torch.cuda.get_device_name(0)} , time:{time.time()-start} \n")

    model_name = 'all-mpnet-base-v2'
    start = time.time()
    num = embedd(t, model_name=model_name) #returns number of embeddings calculated
    with open('time_tracking_2.txt', 'a') as f:
        f.write(f"set: {t}, size:{num} model:{model_name}, gpu:{torch.cuda.get_device_name(0)} , time:{time.time()-start} \n")

    model_name = 'all-MiniLM-L6-v2'
    start = time.time()
    num = embedd( t, model_name=model_name) #returns number of embeddings calculated
    with open('time_tracking_2.txt', 'a') as f:
        f.write(f"set: {t}, size:{num} model:{model_name}, gpu:{torch.cuda.get_device_name(0)} , time:{time.time()-start} \n")

def gpu_umap(table, sql_db):
    
    subr_nums = db_queries.get_all_numbers(table_name=table, sql_db=sql_db)
    print('got ids', len(subr_nums))
    M_embedd = main.fetch_embeddings(num_list = subr_nums, table_name=table, model_name= 'all-MiniLM-L6-v2')
    print('got embeddings', M_embedd.shape)

    coordinates = um.UMAP_to_df_gpu(M_embedd, n_neighbors = 3, n_components=3)
    coordinates.columns = ["x", "y",'z']
    return coordinates

def cluster_coordinates(coordinates):
    scanner = cuml.cluster.hdbscan.HDBSCAN(
        min_cluster_size = 100
    )
    clusters = scanner.fit_predict(coordinates)

    fig_3d = px.scatter_3d(
        coordinates, x='x', y='y', z='z',
        color=clusters, 
        title='UMAP clusters all submissions',
        hover_name=subr_nums
    )

    fig_3d.write_html(f"./clusterall.html")
    return clusters


        

    
if __name__ == '__main__':
    coordinates = gpu_umap('submissions', db_file_s)
    
    clusters = cluster_coordinates(coordinates)

    coordinates["num"] = subr_nums
    coordinates["cluster"] = clusters

    db_file_s.sql("CREATE TABLE umap_coordinates AS SELECT * FROM coordinates")

    db_file_s.sql("CREATE TABLE submission_info AS SELECT * FROM submissions JOIN coordinates ON (coordinates.num = submissions.num); ")


    
    
