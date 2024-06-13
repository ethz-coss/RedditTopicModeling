import pandas as pd
# for gpu version:
import cuml
import config

def PCA_reduction(M_embedd, n_components):
    df = pd.DataFrame(M_embedd)
    features = df.loc[:, :]
    if (config.PCA_COMPONENTS > 0):
        pca_model = cuml.PCA(n_components=config.PCA_COMPONENTS)
        reduced = pca_model.fit_transform(features)
        return reduced
    else: return M_embedd

def UMAP_to_df_gpu(M_embedd, n_neighbors = 2, n_components = 3, min_dist = 0.2):

    reducer = cuml.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist
    )
    print('starting UMAP')
    coordinates = reducer.fit_transform(M_embedd)

    df = pd.DataFrame(coordinates)
    return df



