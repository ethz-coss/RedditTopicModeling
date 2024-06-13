import h5py
import numpy as np
import math
import pandas as pd
from steps import db_queries
import config

#chunck the queries of database and not keep everything in memory
batchsize = 100000


# could pass model or uncomment model assignment
def new_load_embeddings(table_name: str, sql_db, output_file: str, model, device):
    # model = SentenceTransformer("./model/all-MiniLM-L6-v2")

    start = 0
    hf = h5py.File(output_file, 'w')

    lines = db_queries.get_titles(table_name, sql_db, start, 100000)  #load 100K lines from db

    while len(lines) > 0:
        #load and save embeddings in chuncks of 10K (Batchsize)
        embeddings = model.encode(lines, device=device, precision='int8')
        embeddings = np.array(embeddings, dtype=np.int8)
        hf.create_dataset(str(start), data=embeddings)
        print(f'done embedding batch{start}')
        start += len(lines)
        lines = db_queries.get_titles(table_name, sql_db, start, batchsize)

    hf.close()
    print(f'done loading {start} embeddings for {table_name} into {output_file}')
    return start


def embeddings_from_file_id(ids, file_path):
    hf = h5py.File(file_path, 'r')

    tuples = [(str((i // batchsize) * batchsize), i % batchsize) for i in ids]

    tuples = pd.DataFrame(tuples, columns=['batch', 'ids'])
    tuples = tuples.groupby('batch').agg({'ids': lambda x: list(x)})
    tuples = tuples.to_dict()['ids']

    vectors = [hf[i][j] for i, j in tuples.items()]
    M_embedd = np.vstack(vectors)

    return M_embedd
