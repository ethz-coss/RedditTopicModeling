from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def get_tfidf(table_name, sql_db):
    df = sql_db.sql(f"SELECT GROUP_CONCAT(title) AS document, cluster FROM {table_name} "
                    "GROUP BY cluster ORDER BY cluster").fetchdf()
    docs = df['document'].tolist()

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(docs)
    terms = vectorizer.get_feature_names_out()    
    
    return tfidf_matrix, terms

'''
#have to figureout argsort for sparse matrix
def get_top_terms(tfidf_matrix, terms, n=10):  #get_tts
    tf_idf_transposed = tfidf_matrix.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {str(i): [(terms[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1]
                   for i in range(tfidf_matrix.shape[1])}
    return top_n_words
'''

def get_top_terms(tfidf_matrix, terms, n=30):  #get_tts
    top_terms = []

    for i in range(tfidf_matrix.shape[0]):
        row = tfidf_matrix.getrow(i).toarray().flatten()
        top_indices = row.argsort()[-n:][::-1]
        term = [(terms[idx], row[idx]) for idx in top_indices]
        top_terms.append(term)

    return top_terms

#returns clusters ranked by size
def topic_sizes_ranked(table_name, sql_db):
    df = sql_db.sql(f"SELECT COUNT(*) AS sizes, cluster FROM {table_name} GROUP BY cluster ORDER BY sizes DESC").fetchdf()
    return df

#returns clusters in order
def topic_sizes(table_name, sql_db):
    df = sql_db.sql(f"SELECT COUNT(*) AS sizes, cluster FROM {table_name} GROUP BY cluster ORDER BY cluster ASC").fetchdf()
    return df

#returns count of submissions with that term per cluster
def get_terms_count(terms, table_name, sql_db):
    conditions = " OR ".join([f"title LIKE '%{term}%'" for term in terms])

    df = sql_db.sql(f"SELECT COUNT(*) AS count, cluster FROM {table_name}"
                    f" WHERE {conditions} GROUP BY cluster ORDER BY cluster").fetchdf()
    return df

#returns top clusters for a subreddit
def get_subreddit_clusters(subreddit, table_name, sql_db, order = 'count'):
    df = sql_db.sql(f"SELECT cluster, COUNT(*) AS count FROM {table_name} "
                    f"WHERE subreddit = '{subreddit}' "
                    "GROUP BY cluster "
                    f"ORDER BY count DESC").fetchdf()
    return df

#returns top subreddits for a cluster
def get_clusters_subreddit(cluster, table_name, sql_db):
    df = sql_db.sql(f"SELECT subreddit, COUNT(*) AS count FROM {table_name} "
                    f"WHERE cluster = {cluster} "
                    "GROUP BY subreddit "
                    "ORDER BY count DESC").fetchdf()
    return df

def get_term_clusters(terms, table_name, sql_db):
    conditions = " OR ".join([f"title LIKE '%{term}%'" for term in terms])

    df = sql_db.sql(f"SELECT COUNT(*) AS count, cluster FROM {table_name}"
                    f" WHERE {conditions} GROUP BY cluster ORDER BY count DESC").fetchdf()
    return df