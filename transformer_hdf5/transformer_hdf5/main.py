import time
import duckdb
import load_to_db
import embeddings
import db_queries
import torch
from sentence_transformers import SentenceTransformer


subr = ['Republican', 'democrats', 'healthcare', 'Feminism', 'nra', 'education', 'climatechange',
        'politics', 'progressive', 'The_Donald', 'TrueChristian', 'Trucks', 'teenagers', 'AskMenOver30',
        'backpacking', 'news', 'BlackLivesMatter', 'racism', 'news', 'usa', 'DefundPoliceNYC']

# this is the file to which the sql database will be saved
db_file_c = duckdb.connect("/cluster/work/coss/anmusso/victoria/loaded_data/loaded_comments.db")
db_file_s = duckdb.connect("/cluster/work/coss/anmusso/victoria/loaded_data/loaded_submissions.db")

#adjust file basepath
def load(C_or_S, month, year):
    #load comments into database
    tablename = f'{C_or_S}_{month}_{year}'

    start = time.time()
    if C_or_S == 'C':
        sql_db = db_file_c
        input_file = f'/cluster/work/coss/anmusso/reddit/comments/RC_{year}-{month}.zst'
        load_to_db.extract_comments(input_file_name=input_file, table_name=tablename, sql_db=sql_db)
    else:
        sql_db = db_file_s
        input_file = f'/cluster/work/coss/anmusso/reddit/submissions/RS_{year}-{month}.zst'
        load_to_db.extract_submissions(input_file_name=input_file, table_name=tablename, sql_db=sql_db)
    
    
    print('time for loading:', time.time() - start)


#adjust file basepaths
def embedd(month, year, C_or_S, model_name = 'all-MiniLM-L6-v2'):
    model = SentenceTransformer(f"/cluster/work/coss/anmusso/victoria/model/{model_name}")
    tablename = table_id(C_or_S, month, year)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if C_or_S == 'C':
        sql_db = db_file_c
    else:
        sql_db = db_file_s

    embeddings_file = f'./embeddings_{tablename}_{model_name}.h5py'
    start = time.time()
    num = embeddings.new_load_embeddings(table_name=tablename, sql_db=sql_db, output_file=embeddings_file, model=model, device = device)
    print('time for embedding:', time.time() - start)
    return num

def fetch_embeddings(num_list, table_name, model_name):
    file_path = f"/cluster/work/coss/anmusso/victoria/embeddings/embeddings_{table_name}_{model_name}.h5py"
    return embeddings.embeddings_from_file_id(num_list, file_path)

def table_id(C_or_S, month, year):
    return f'{C_or_S}_{month}_{year}'

if __name__ == '__main__':
  
    t,m,y = 'S','05','2020' #type month and year
    #model_name = 'all-mpnet-base-v2'
    model_name = 'all-MiniLM-L6-v2'

    start = time.time()
    num = embedd(m, y, t, model_name=model_name) #returns number of embeddings calculated
    with open('time_tracking_2.txt', 'a') as f:
        f.write(f"set: {m+t+y}, size:{num} model:{model_name}, gpu:{torch.cuda.get_device_name(0)} , time:{time.time()-start} \n")

  
    '''
    start = time.time()
    device = torch.device("cpu")
    embedd('05', '2020', 'S', device=device)
    with open('time_tracking.txt', 'a') as f:
        f.write(f"set: 05_2020_S, model:all-MiniLM-L6-v2, gpu:none ,time:{time.time()-start}")
    '''
    
