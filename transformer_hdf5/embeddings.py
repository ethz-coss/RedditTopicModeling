import h5py
import numpy as np
import db_queries
import math

#chunck the queries of database and not keep everything in memory
batchsize = 10000


# could pass model or uncomment model assignment
def new_load_embeddings(table_name: str, sql_db, output_file: str, model, device):
    # model = SentenceTransformer("./model/all-MiniLM-L6-v2")

    start = 0
    hf = h5py.File(f'/cluster/work/coss/anmusso/victoria/embeddings/{output_file}', 'w')

    lines = db_queries.get_titles(table_name, sql_db, start, 100000) #load 100K lines from db
    
    while len(lines)>0:   
        for i in range(math.ceil(len(lines)/ batchsize)): 
            #batch.to(device)
            batch = lines[i * batchsize:(i + 1) * batchsize] #load and save embeddings in chuncks of 10K (Batchsize)
            embeddings = model.encode(batch, device = device)
            embeddings = np.array(embeddings, dtype=np.float64)
            hf.create_dataset(str(start), data=embeddings)
            print(f'done embedding batch{start}')
            start += len(batch)

        lines = db_queries.get_titles(table_name, sql_db, start, batchsize)

    hf.close()
    print(f'done loading {start} embeddings for {table_name} into {output_file}')
    return start


def embeddings_from_file_id(ids, file_path ):
    embeddings = []
    hf = h5py.File(file_path, 'r')

    i = 0
    while i < len(ids):
        batch_id = ids[i] // batchsize  #only extract needed batches from file
        data_batch = hf[str(batch_id*batchsize)]  #load batch of 1K embeddings
        #print(batch_id)
        while i < len(ids) and ids[i] // batchsize == batch_id:  #extract the needed ones
            embeddings.append(data_batch[ids[i] % batchsize])
            i += 1

    M_embedd = np.matrix(embeddings)

    return M_embedd
