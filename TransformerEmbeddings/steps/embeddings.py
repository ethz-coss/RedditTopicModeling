import pandas as pd
import numpy as np
import h5py
from sentence_transformers.quantization import quantize_embeddings

from steps import db_queries

BATCHSIZE = 100000


# could pass model or uncomment model assignment
def new_load_embeddings(table_name: str, sql_db, output_file: str, model, device):
    # model = SentenceTransformer("./model/all-MiniLM-L6-v2")

    start = 0
    hf = h5py.File(output_file, 'w')

    lines = db_queries.get_titles(table_name, sql_db, start, 100000)  #load 100K lines from db

    while len(lines) > 0:
        #load and save embeddings in chuncks of 10K (Batchsize)
        embeddings = model.encode(lines, device=device)
        embeddings = np.array(embeddings)
        hf.create_dataset(str(start), data=embeddings)
        print(f'done embedding batch{start}', type(embeddings))
        start += len(lines)
        lines = db_queries.get_titles(table_name, sql_db, start, BATCHSIZE)

    hf.close()
    print(f'done loading {start} embeddings for {table_name} into {output_file}')
    return start



def embeddings_from_file_id(ids, file_path):
    hf = h5py.File(file_path, 'r')

    tuples = [((i // BATCHSIZE) * BATCHSIZE, i % BATCHSIZE) for i in ids]

    tuples = pd.DataFrame(tuples, columns=['batch', 'ids'])
    tuples = tuples.groupby('batch').agg({'ids': lambda x: list(x)})
    tuples = tuples.to_dict()['ids']

    vectors = [hf[str(i)][j] for i, j in tuples.items()]
    M_embedd = np.vstack(vectors)

    return M_embedd

def quantize(input_file, output_file, precision = 'int8', new_batchsize = 100_000, old_batchsize = BATCHSIZE):
    hf = input_file

    for i in range(0, len(hf.keys()), 10):
        vectors = np.matrix(hf[str(i * old_batchsize)][:])
        for j in range(1, new_batchsize//old_batchsize):
            if i + j < len(hf.keys()):
                ds = np.matrix(hf[str((i + j) * old_batchsize)][:])
                vectors = np.vstack((vectors, ds))
                del ds
        small = quantize_embeddings(vectors, precision=precision)
        del vectors
        output_file.create_dataset(str(i * old_batchsize), data=small)

    print('finished quantizing embeddings')


