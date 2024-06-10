import pandas as pd
# for gpu version:
import cuml



def UMAP_to_df_gpu(M_embedd, n_neighbors = 2, n_components = 3, min_dist = 0.2):
    EMBEDDING_DIM = M_embedd.shape[1]  # for transformers Model
    df = pd.DataFrame(M_embedd)
    features = df.loc[:, :]

    reducer = cuml.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist
    )
    coordinates = reducer.fit_transform(features)

    df = pd.DataFrame(coordinates)
    return df



