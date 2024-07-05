import config
import duckdb
import numpy as np
from collections import Counter
import pandas as pd
import steps.topic_finding as tf
import steps.plotting as plot

sql_db = duckdb.connect('trydb.db')

BATCHSIZE = 100


def try_sql():
    sql_db.sql("CREATE TABLE chaos (NUM INT PRIMARY KEY , title VARCHAR)")
    sql_db.sql("INSERT INTO chaos VALUES (2, 'hello2')")
    sql_db.sql("INSERT INTO chaos VALUES (1, 'hello1')")
    sql_db.sql("INSERT INTO chaos VALUES (0, 'hello0')")
    df = sql_db.sql("SELECT title FROM chaos ORDER BY num").fetchdf()

    print(df)


def try_order():
    ids = [i for i in range(0, 2_000, 10)]

    tuples = [((i // BATCHSIZE) * BATCHSIZE, i % BATCHSIZE) for i in ids]

    tuples = pd.DataFrame(tuples, columns=['batch', 'ids'])
    tuples = tuples.groupby('batch').agg({'ids': lambda x: list(x)})
    tuples = tuples.to_dict()['ids']
    print(type(tuples))
    print(tuples)

    last = -1
    for i, vectors in tuples.items():
        if int(i) < last: print('i not sorted')
        if all(vectors[i] <= vectors[i + 1] for i in range(len(vectors) - 1)):
            last = last
        else:
            print("No, the list is not sorted.")


def do_plots(num):
    duck_database = duckdb.connect(config.DATA_BASE_PATH)

    info_table = 'submissions_info_filtered'

    tfidf, terms = tf.get_tfidf(info_table, duck_database)
    top_words = tf.get_top_terms(tfidf, terms)
    sizes = tf.topic_sizes(info_table, duck_database)
    sizes = sizes['sizes'].tolist()

    candidates = [num]

    for n in candidates:
        print('top cluster: ', n)
        print('size of top cluster: ', sizes[n + 1])
        print('top words of top cluster: ', top_words[n + 1])
        print('top subreddits of top cluster: ', tf.get_clusters_subreddit(n, info_table, duck_database))
        plot.unique_per_day(info_table, duck_database, n)


if __name__ == '__main__':
    do_plots(1170)
