import config
import duckdb
import numpy as np
from collections import Counter
import steps.topic_finding as tf
import steps.plotting as plot


def do_plots():

    duck_database = duckdb.connect(config.DATA_BASE_PATH)

    info_table = 'submissions_info_filtered'

    tfidf, terms = tf.get_tfidf(info_table, duck_database)
    top_words = tf.get_top_terms(tfidf, terms)
    sizes = tf.topic_sizes(info_table, duck_database)
    sizes = sizes['sizes'].tolist()


    candidates = [525]

    for n in candidates:
        print('top cluster: ', n)
        print('size of top cluster: ', sizes[n + 1])
        print('top words of top cluster: ', top_words[n + 1])
        print('top subreddits of top cluster: ', tf.get_clusters_subreddit(n, info_table, duck_database))
        plot.submissions_timeline(info_table,duck_database,n)
        plot.author_timeline(info_table, duck_database, n)
        plot.subreddit_timeline(info_table, duck_database, n)
        plot.top_subreddit_timeline(info_table, duck_database, n)
        plot.submissions_per_day(info_table, duck_database, n)


if __name__ == '__main__':
    do_plots()