import numpy as np
import pandas as pd
import cuml
import config
import steps.embeddings as embeddings
import gc
#import rmm


def UMAP_embeddings(subr_nums, file_path):
    if config.BATCH_UMAP:
        coordinates = batched_UMAP(subr_nums=subr_nums, n_neighbors=config.UMAP_N_Neighbors,
                                    n_components=config.UMAP_COMPONENTS, min_dist=config.UMAP_MINDIST)
    else:

        M_embedd = embeddings.embeddings_from_file_id(subr_nums, file_path)
        print('done retrieving embeddings', M_embedd.shape)
        coordinates = UMAP_to_df_gpu(M_embedd, n_neighbors=config.UMAP_N_Neighbors,
                                        n_components=config.UMAP_COMPONENTS, min_dist=config.UMAP_MINDIST)
        coordinates.columns = [str(i) for i in range(config.UMAP_COMPONENTS)]
        del M_embedd

    return coordinates

def PCA_reduction(M_embedd):
    df = pd.DataFrame(M_embedd)
    features = df.loc[:, :]
    if (config.PCA_COMPONENTS > 0):
        pca_model = cuml.PCA(n_components=config.PCA_COMPONENTS)
        reduced = pca_model.fit_transform(features)
        return reduced
    else:
        return M_embedd


def UMAP_to_df_gpu(M_embedd, n_neighbors=2, n_components=3, min_dist=0.2):
    reducer = cuml.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist
    )
    coordinates = reducer.fit_transform(M_embedd)

    df = pd.DataFrame(coordinates)
    return df


def batched_UMAP(subr_nums, n_neighbors=2, n_components=3, min_dist=0.2):

    shuffled_nums = np.random.permutation(subr_nums)
    df = pd.DataFrame(columns=[str(i) for i in range(config.UMAP_COMPONENTS)])

    reducer = cuml.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist
    )

    BS = config.UMAP_BATCHSIZE
    file_path = config.EMBEDDINGS_FILE
    sorted_nums = sorted(shuffled_nums[0:BS])
    vectors = embeddings.embeddings_from_file_id(sorted_nums, file_path)
    reducer.fit(vectors)

    for i in range(0, len(shuffled_nums) // BS, BS):

        del vectors
        gc.collect()
        #rmm.reinitialize()

        sorted_nums = sorted(shuffled_nums[i:i + BS])
        vectors = embeddings.embeddings_from_file_id(sorted_nums, file_path)
        coordinates = reducer.transform(vectors)

        for j, idx in enumerate(sorted_nums):
            df.loc[idx] = coordinates[j]

    return df
