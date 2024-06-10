import plotly.express as px
import plotly.graph_objects as go
import time


#used during embedding, default returns up to 100Mil
def get_titles(table_name, sql_con, start=0, batchsize=100000000):
    df = sql_con.sql(f'SELECT title from {table_name} WHERE num >= {start} and num < {start+batchsize} ;').fetchdf()
    return df['title'].tolist()


def get_subreddit_numbers(table_name, sql_db, subreddits):
     subr_list_str = ','.join([f"'{sub}'" for sub in subreddits])
     l = sql_db.sql(f"SELECT num FROM {table_name} WHERE subreddit IN ({subr_list_str}) ").fetchnumpy()
     return l['NUM'].tolist()

def get_all_numbers(table_name, sql_db):
    df = sql_db.sql(f"SELECT num FROM {table_name}").fetchnumpy()
    return df['NUM'].tolist()

def get_filtered_submissions(table_name, sql_db):
    df = sql_db.sql(f"SELECT num FROM {table_name} WHERE score > 20 AND num_comments > 10").fetchnumpy()
    return df['NUM'].tolist()



