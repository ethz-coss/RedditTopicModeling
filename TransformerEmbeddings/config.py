
MONTHS_COMMENTS = []
MONTHS_SUBMISSIONS = ['05', '06']

EMBEDD_SET = ['S']
ANALYSE_SET = 'S'

INPUT_FILE_BASE_PATH = '/cluster/work/coss/anmusso/reddit'
EMBEDDINGS_BASE_PATH = '/cluster/work/coss/anmusso/victoria/embeddings/'

#adjust these 3 parameters
MODEL_PATH = f'/cluster/work/coss/anmusso/victoria/model/all-MiniLM-L6-v2'
EMBEDDINGS_FILE = '/cluster/work/coss/anmusso/victoria/embeddings/embeddings_submissions_small.h5py' #where embeddingfiles will be saved
DATA_BASE_PATH = '/cluster/work/coss/anmusso/victoria/loaded_data/loaded_data.db'


FILTER_UMAP= False
PCA_COMPONENTS = 50
UMAP_COMPONENTS = 5
UMAP_N_Neighbors = 50
UMAP_MINDIST = 0.1

HDBS_MIN_CLUSTERSIZE= 100
#HDBS_ALG = ''

TERMS = ['blm', 'Black Lives Matter' ,'black lives matter', 'blacklivesmatter', 'BlackLivesMatter', 'BLM', 'racism', 'George', 'Floyd' ,]
SUBREDDIT = 'BlackLivesMatter'
