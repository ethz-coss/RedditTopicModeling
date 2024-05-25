import pandas as pd
import umap.plot
import plotly.express as px

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


    
#other UMap method
'''
#umap_2d = umap.UMAP(n_components=2, init='random', random_state=1, n_neighbors=5)
umap_2d = umap.UMAP()
proj_2d = umap_2d.fit(features)

umap.plot.points(proj_2d, labels=df.source)

hover_data = pd.DataFrame({'index': np.arange(len(titles)),
                        'label': df.source,
                        'title': titles})

p = umap.plot.interactive(proj_2d, labels=df.source, hover_data=hover_data )

umap.plot.show(p)
'''
