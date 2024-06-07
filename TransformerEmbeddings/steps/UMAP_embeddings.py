import pandas as pd
import umap.plot
import plotly.express as px

# for gpu version:
import cuml



def show_umap(M_embedd, meta, output_file):

    EMBEDDING_DIM = M_embedd.shape[1]  # for transformers Model
    titles = meta['title'].tolist()
    subreddits = meta['subreddit'].tolist()

    # create dataframe
    df = pd.DataFrame(M_embedd)
    df = df.assign(source=subreddits)
    features = df.loc[:, :(EMBEDDING_DIM - 1)]

    # create mapping, start diagramm
    umap_2d = umap.UMAP(n_neighbors=5)
    proj_2d = umap_2d.fit_transform(features)
    print('transformed and fit')

    fig_2d = px.scatter(
        proj_2d, x=0, y=1, 
        color=df.source, labels={'color': 'source'},
        title='UMAP comments 20_05',
        hover_name=titles
    )

    fig_2d.write_html(f"./{output_file}.html")


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


def show_umap_2d(coordinates, color, title, hover_data=None):
    fig_2d = px.scatter(
        coordinates, x='x', y='y',
        color=color,
        title= title,
        hover_name=hover_data
    )

    fig_2d.update_traces(marker=dict(size=2, opacity=0.5))
    fig_2d.write_html(f"./{title}.html")
    del fig_2d
    
def show_umap_3d(coordinates, color, title, hover_data=None):
    fig_3d = px.scatter_3d(
        coordinates, x='x', y='y', z='z',
        color=color,
        title='UMAP clusters 500k submission',
        hover_name=hover_data
    )

    fig_3d.update_traces(marker=dict(size=2, opacity=0.5))
    fig_3d.write_html(f"./cluster500k_small.html")
    del fig_3d

