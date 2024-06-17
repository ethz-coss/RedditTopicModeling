import h5py
import numpy as np


old_batchsize = 10000
new_batchsize = 100_000


hf = h5py.File('/cluster/work/coss/anmusso/victoria/embeddings/embeddings_submissions_small.h5py', 'r')

keys = [int(key) for key in hf.keys()]
keys = sorted(keys)

print(keys)

