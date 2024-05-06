import pandas as pd
import load_embeddings
import umap
import umap.plot
import plotly.express as px
import numpy as np
import duck_try


source = 'vectors_C_20_05.h5'

#data1 = ["democrats", "Feminism"]
M_embedd, meta = load_embeddings.embeddings_from_file(source) #adjust file path in function

titles = [meta[0][i] for i in range(len(meta[0]))]

# create dataframe
df = pd.DataFrame(M_embedd)
df = df.assign(source=[meta[1][i] for i in range(len(meta[1]))])
features = df.loc[:, :(reddit_projection.EMBEDDING_DIM - 1)]


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
# create mapping, start diagramm
umap_2d = umap.UMAP(n_components=2, init='random', random_state=0)
proj_2d = umap_2d.fit_transform(features)
# proj_3d = umap_3d.fit_transform(features)
print('transformed and fit')

fig_2d = px.scatter(
    proj_2d, x=0, y=1,
    color=df.source, labels={'color': 'source'},
    title='UMAP comments 20_05',
    hover_name=titles
)

fig_2d.show()
'''

