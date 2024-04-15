import pandas as pd
import example
import reddit_projection
import projection
import umap
import plotly.express as px

ref = "./data/old"
collection_name = "pdfs"

collection = example.chroma_client.get_collection(collection_name)
print("collection count: ", collection.count())

#M_embedd, meta = projection.new_load_embeddings_n(source=ref, collection_name=collection_name)

M_embedd, meta = projection.embeddings_from_collection(collection_name=collection_name)




df = pd.DataFrame(M_embedd)
#print(df)
df = df.assign(source=[meta[i]['source'] for i in range(len(meta))])
features = df.loc[:, :(reddit_projection.EMBEDDING_DIM-1)]

#print(features)

umap_2d = umap.UMAP(n_components=2, init='random', random_state=0,n_neighbors=10)
proj_2d = umap_2d.fit_transform(features)
print('transformed and fit')

fig_2d = px.scatter(
    proj_2d, x=0, y=1,
    color=df.source, labels={'color': 'source'},
    title = 'Pdfs UMAP N=10',
    hover_name=[meta[i]['text'] for i in range(len(meta))] 
)

fig_2d.show()



umap_2d = umap.UMAP(n_components=2, init='random', random_state=0,n_neighbors=50)
proj_2d = umap_2d.fit_transform(features)
print('transformed and fit')

fig_2d = px.scatter(
    proj_2d, x=0, y=1,
    color=df.source, labels={'color': 'source'},
    title = 'Pdfs UMAP N=50',
    hover_name=[meta[i]['text'] for i in range(len(meta))] 
)

fig_2d.show()

#fig_3d = px.scatter_3d(
#   proj_3d, x=0, y=1, z=2,
#    color=df.source, labels={'color': 'source'}
#)
#fig_3d.update_traces(marker_size=5)


#fig_3d.show()
