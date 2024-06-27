import config
import duckdb
import numpy as np
from collections import Counter
import steps.topic_finding as tf

duck_database = duckdb.connect(config.DATA_BASE_PATH)

info_table = 'submissions_info_filtered'

tfidf, terms = tf.get_tfidf(info_table, duck_database)
top_words = tf.get_top_terms(tfidf, terms)

relevant = [ 'blm', 'blacklivesmatter', 'BlackLivesMatter', 'BLM', 'racism', 'George', 'Floyd']
indx = []  # indexes or terms we want to look at

for word in relevant:
    index = np.where(terms == word)[0]  # np.where returns a tuple, the first element is an array of indices
    if index.size > 0:
        indx.append(index[0])  # Append the first occurrence

clusters = []
for i in indx:
    col = tfidf.getcol(i).toarray().flatten()
    top_indices = col.argsort()[-10:][::-1]
    for t in top_indices: clusters.append(t)

print(clusters)

count = Counter(clusters)

# Get the 5 most common numbers
most_common_numbers = count.most_common(10)
print(most_common_numbers)
# Extract just the numbers (without counts)
top_5_numbers = [num for num, _ in most_common_numbers]

sizes = tf.topic_sizes(info_table, duck_database)
sizes = sizes['sizes'].tolist()

for n in top_5_numbers:
    print('top cluster: ', n)
    print('size of top cluster: ', sizes[n + 1])
    print('top words of top cluster: ', top_words[n + 1])
    print('top subreddits of top cluster: ', tf.get_clusters_subreddit(n, info_table, duck_database))



