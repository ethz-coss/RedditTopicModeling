import pandas as pd
from sklearn.decomposition import PCA
import example
from load_reddit_data import reddit_projection
import plotly.express as px

collection_name = "Reddit-Comments"
data = ['Republican', 'democrats', 'healthcare', 'Feminism', 'nra', 'education', 'climatechange', 'politics',
        'random', 'teenagers', 'progressive', 'The_Donald', 'TrueChristian', 'Trucks', 'AskMenOver30',
        'backpacking']

collection = example.chroma_client.get_collection(collection_name)
print(collection.count())

M_embedd, meta = reddit_projection.embeddings_from_collection(collection_name=collection_name, subreddits=data)
print(len(M_embedd))
print(len(meta))

df = pd.DataFrame(M_embedd)
print(df)
df = df.assign(source=[meta[i]['subreddit'] for i in range(len(meta))])
features = df.loc[:, :(reddit_projection.EMBEDDING_DIM - 1)]

pca = PCA(n_components=2)
PrincipalComponents = pca.fit_transform(features)

fig_2d = px.scatter(
    PrincipalComponents, x=0, y=1,
    color=df.source, labels={'color': 'source'}
)
fig_2d.show()
