import pandas as pd
import example
import reddit_projection
import umap
import plotly.express as px
import numpy as np
import duckdb_store


collection_name = "Reddit-Comments-2"
collection = example.chroma_client.get_collection(collection_name)
print(collection.count())


df = duckdb_store.get_good_ids() #change to filter the right entries in the database
ids = list(df['id'])

#retrieve embeddings for posts within the subreddits
data1 = ['Republican', 'democrats', 'healthcare', 'Feminism', 'nra', 'education', 'progressive', 'The_Donald']
M_embedd, meta = reddit_projection.embeddings_from_collection_id(collection_name, ids=ids, subreddits=data1)
titles = [meta[i]['title'] for i in range(len(meta))]


# create dataframe
df = pd.DataFrame(M_embedd)
df = df.assign(source=[meta[i]['subreddit'] for i in range(len(meta))])
features = df.loc[:, :(reddit_projection.EMBEDDING_DIM - 1)]

# create mapping, start diagramm
umap_2d = umap.UMAP(n_components=2, init='random', random_state=0, n_neighbors=30)
proj_2d = umap_2d.fit_transform(features)
# proj_3d = umap_3d.fit_transform(features)
print('transformed and fit')

fig_2d = px.scatter(
    proj_2d, x=0, y=1,
    color=df.source, labels={'color': 'source'},
    title='UMAP full collection N=30',
    hover_name=titles
)

fig_2d.show()

umap_2d = umap.UMAP(n_components=2, init='random', random_state=0, n_neighbors=5)
proj_2d = umap_2d.fit_transform(features)
fig_2d = px.scatter(
    proj_2d, x=0, y=1,
    color=df.source, labels={'color': 'source'},
    title='UMAP full collection N=5',
    hover_name=titles
)

fig_2d.show()
