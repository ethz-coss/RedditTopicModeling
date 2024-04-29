import pandas as pd
import reddit_projection
import umap
import umap.plot
import plotly.express as px
import numpy as np
import duck_try

#lines = reddit_projection.example_read_data('C:/Users/victo/PycharmProjects/RedditProject/data/RS_2020-06_filtered.zst')
#id_dict = dict((item['id'], item) for item in lines)
# print(meta[0])
# print (id_dict)
#titles = [id_dict[meta[i]['id']]['title'] for i in range(len(meta))]

source = 'C:/Users/victo/PycharmProjects/RedditProject/transformer_hdf5/data/RS_2020-06_filtered.zst'
reddit_projection.new_load_embeddings_t(source=source)

df = duck_try.get_good_ids()
titles1 = list(df['title'])
ids = list(df['id'])

data1 = ['Republican', 'democrats', 'healthcare', 'Feminism', 'nra', 'education', 'progressive', 'The_Donald']
#data1 = ["democrats", "Feminism"]
M_embedd, meta = reddit_projection.embeddings_from_file_id(ids)
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

'''# create mapping, start diagramm
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
'''
