import pandas as pd
import example
from load_reddit_data import reddit_projection

import umap
import plotly.express as px


collection_name = "Reddit-Comments"
data = ['Republican', 'democrats', 'healthcare', 'Feminism', 'nra', 'education', 'climatechange', 'politics',
        'random','teenagers', 'progressive', 'The_Donald', 'TrueChristian', 'Trucks', 'AskMenOver30',
        'backpacking']

collection = example.chroma_client.get_collection(collection_name)
print(collection.count())

M_embedd, meta = reddit_projection.embeddings_from_collection(collection_name=collection_name, subreddits=data)
print(len(M_embedd))
print(len(meta))

df = pd.DataFrame(M_embedd)
print(df)
df = df.assign(source=[meta[i]['subreddit'] for i in range(len(meta))])
features = df.loc[:, :reddit_projection.EMBEDDING_DIM]

print(features)

umap_2d = umap.UMAP(n_components=2, init='random', random_state=0)
umap_3d = umap.UMAP(n_components=3, init='random', random_state=0)
print('initialized')

proj_2d = umap_2d.fit_transform(features)
proj_3d = umap_3d.fit_transform(features)
print('transformed and fit')

fig_2d = px.scatter(
    proj_2d, x=0, y=1,
    color=df.source, labels={'color': 'source'}
)
fig_3d = px.scatter_3d(
   proj_3d, x=0, y=1, z=2,
   color=df.source, labels={'color': 'source'}
)
fig_3d.update_traces(marker_size=5)

fig_2d.show()
fig_3d.show()
