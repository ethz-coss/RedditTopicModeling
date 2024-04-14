import pandas as pd
import example
from load_reddit_data import reddit_projection, projection
import umap
import plotly.express as px

ref = "./data/old"
collection_name = "pdfs"

M_embedd, meta = projection.new_load_embeddings_n(source=ref, collection_name=collection_name)
#M_embedd, meta = projection.embeddings_from_collection(collection_name=collection_name)

collection = example.chroma_client.get_collection(collection_name)
print(collection.count())

df = pd.DataFrame(M_embedd)
print(df)
df = df.assign(source=[meta[i]['source'] for i in range(len(meta))])
features = df.loc[:, :(reddit_projection.EMBEDDING_DIM-1)]

print(features)

umap_2d = umap.UMAP(n_components=2, init='random', random_state=0)
#umap_3d = umap.UMAP(n_components=3, init='random', random_state=0)
print('initialized')

proj_2d = umap_2d.fit_transform(features)
#proj_3d = umap_3d.fit_transform(features)
print('transformed and fit')

fig_2d = px.scatter(
    proj_2d, x=0, y=1,
    color=df.source, labels={'color': 'source'},
    hover_name=[meta[i]['text'] for i in range(len(meta))] #change score to text
)

#fig_3d = px.scatter_3d(
#   proj_3d, x=0, y=1, z=2,
#    color=df.source, labels={'color': 'source'}
#)
#fig_3d.update_traces(marker_size=5)

fig_2d.show()
#fig_3d.show()
