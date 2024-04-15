import pandas as pd
import example
import reddit_projection

import umap
import plotly.express as px


collection_name = "Reddit-Comments-2"
'''
data = ['Republican', 'democrats', 'healthcare', 'Feminism', 'nra', 'education', 'climatechange', 'politics',
        'random', 'progressive', 'The_Donald', 'TrueChristian', 'Trucks', 'AskMenOver30',
        'backpacking']
'''
data1 = ['climatechange', 'Feminism', 'democrats', 'Republican']
data2 = ['healthcare', 'politics', 'AskMenOver30', 'nra']

collection = example.chroma_client.get_collection(collection_name)
print('collection count: ', collection.count())

#retrieve embeddings
M_embedd, meta = reddit_projection.embeddings_from_collection(collection_name=collection_name, subreddits=data1)

#create dataframe
df = pd.DataFrame(M_embedd)
df = df.assign(source=[meta[i]['subreddit'] for i in range(len(meta))])
features = df.loc[:, :(reddit_projection.EMBEDDING_DIM-1)]

# create mapping, start diagramm
umap_2d = umap.UMAP(n_components=2, init='random', random_state=0,n_neighbors=30)
proj_2d = umap_2d.fit_transform(features)
fig_2d = px.scatter(
    proj_2d, x=0, y=1,
    color=df.source, labels={'color': 'source'},
    title = 'UMAP full collection N=30'
)

fig_2d.show()

umap_2d = umap.UMAP(n_components=2, init='random', random_state=0,n_neighbors=5)
proj_2d = umap_2d.fit_transform(features)
fig_2d = px.scatter(
    proj_2d, x=0, y=1,
    color=df.source, labels={'color': 'source'},
    title = 'UMAP full collection N=50'
)

fig_2d.show()

