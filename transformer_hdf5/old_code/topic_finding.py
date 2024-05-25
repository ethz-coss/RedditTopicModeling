from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def get_tfidfs(table_name, sql_db):
    df = sql_db.sql(f"SELECT GROUP_CONCAT(title [SEPERATOR ' ']) AS document, cluster FROM {table_name} "
                    "GROUP BY cluster ORDER BY cluster").fetchdf()
    docs = df['document'].tolist()

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(docs)
    terms = vectorizer.get_feature_names_out()

    return tfidf_matrix, terms


def get_top_terms(tfidf_matrix, terms, n=10):  #get_tts
    tf_idf_transposed = tfidf_matrix.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {str(i): [(terms[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1]
                   for i in range(tfidf_matrix.shape[1])}
    return top_n_words


def get_topic_sizes(table_name, sql_db):
    df = sql_db.sql(f"SELECT COUNT(*) AS sizes FROM {table_name} GROUP BY cluster ORDER BY cluster").fetchdf()
    return df['sizes'].tolist()


def get_terms_count(terms, table_name, sql_db):
    conditions = " OR ".join([f"title LIKE '%{term}%'" for term in terms])

    df = sql_db.sql(f"SELECT COUNT(*) AS count, cluster FROM {table_name}"
                    f" WHERE {conditions} GROUP BY cluster ORDER BY cluster").fetchdf()
    return df


def get_subreddit_clusters(subreddit, table_name, sql_db):
    df = sql_db.sql(f"SELECT cluster, COUNT(*) AS count FROM {table_name}"
                    f"WHERE subreddit = '{subreddit}' "
                    "GROUP BY cluster "
                    "ORDER BY count DESC").fetchdf()
    return df