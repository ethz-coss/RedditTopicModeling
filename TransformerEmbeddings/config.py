
C_or_S = 'S'
MONTHS = ['05', '06']


INPUT_FILE_BASE_PATH = '/cluster/work/coss/anmusso/reddit'
EMBEDDINGS_BASE_PATH = '/cluster/work/coss/anmusso/victoria/embeddings/'

#adjust these 3 parameters
MODEL_PATH = f'/cluster/work/coss/anmusso/victoria/model/all-MiniLM-L6-v2'
EMBEDDINGS_FILE = '/cluster/work/coss/anmusso/victoria/embeddings/embeddings_eng_all-MiniLM-L6-v2_filtered.h5py' #where embeddingfiles will be saved
COORDINATES_FILE = '/cluster/work/coss/anmusso/victoria/coordinates/submissions_coordinates_eng_filtered.parquet'
DATA_BASE_PATH = '/cluster/work/coss/anmusso/victoria/loaded_data/loaded_data_eng_filtered.db'


FILTER = True
BATCH_UMAP = False

UMAP_COMPONENTS = 5
UMAP_N_Neighbors = 60
UMAP_MINDIST = 0.01
UMAP_BATCHSIZE =4_000_000


HDBS_MIN_CLUSTERSIZE= 100
#HDBS_MIN_SAMPLES = 100

PCA_COMPONENTS = 50

#HDBS_ALG = ''


TERMS = ['blm', 'Black Lives Matter', 'black lives matter', 'blacklivesmatter', 'BlackLivesMatter',
         'BLM', 'racism', 'George', 'Floyd', 'police', 'protest']
SUBREDDIT = 'BlackLivesMatter'
