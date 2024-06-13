import numpy as np
import h5py
from sentence_transformers.quantization import quantize_embeddings
import wrappers
print('imported')

hf = h5py.File('/cluster/work/coss/anmusso/victoria/embeddings/embeddings_submissions_all-MiniLM-L6-v2.h5py', 'r')

outfile = h5py.File('/cluster/work/coss/anmusso/victoria/embeddings/embeddings_submissions_small.h5py', 'w')

old_batchsize = 10000
new_batchsize = 100_000


for i in range(0, len(hf.keys()), 10):
    vectors = np.matrix(hf[str(i*old_batchsize)][:])
    for j in range(1, 10):
        if i+j < len(hf.keys()):
            ds = np.matrix(hf[str((i+j)*old_batchsize)][:])
            vectors = np.vstack((vectors, ds))
            del ds
    small = quantize_embeddings(vectors, precision='int8')
    del vectors
    outfile.create_dataset(str(i * old_batchsize), data=small)

print('finished quantizing embeddings')


