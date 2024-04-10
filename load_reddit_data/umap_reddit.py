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
data1 = ['climatechange', 'Feminism', 'democrats', 'republican']
data2 = ['healthcare', 'politics', 'AskMenOver30', 'nra']

collection = example.chroma_client.get_collection(collection_name)
print('collection count: ', collection.count())

#retrieve embeddings
M_embedd1, meta1 = reddit_projection.embeddings_from_collection(collection_name=collection_name, subreddits=data1)

#create dataframe
df = pd.DataFrame(M_embedd1)
print(df)
df = df.assign(source=[meta1[i]['subreddit'] for i in range(len(meta1))])
features = df.loc[:, :reddit_projection.EMBEDDING_DIM]

# create mapping, start diagramm
umap_2d = umap.UMAP(n_components=2, init='random', random_state=0)
proj_2d = umap_2d.fit_transform(features)
fig_2d = px.scatter(
    proj_2d, x=0, y=1,
    color=df.source, labels={'color': 'source'}
)


# 3d version for data 2
M_embedd2, meta2 = reddit_projection.embeddings_from_collection(collection_name=collection_name, subreddits=data2)
df = pd.DataFrame(M_embedd2)
print(df)
df = df.assign(source=[meta2[i]['subreddit'] for i in range(len(meta2))])
features = df.loc[:, :reddit_projection.EMBEDDING_DIM]

umap_3d = umap.UMAP(n_components=3, init='random', random_state=0)
print('initialized')
proj_3d = umap_3d.fit_transform(features)
print('transformed and fit')

fig_3d = px.scatter_3d(
   proj_3d, x=0, y=1, z=2,
   color=df.source, labels={'color': 'source'}
)
fig_3d.update_traces(marker_size=5)


#show both diagramms
fig_2d.show()
fig_3d.show()
