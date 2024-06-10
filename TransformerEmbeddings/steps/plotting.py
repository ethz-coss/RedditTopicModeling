import pandas as pd
import plotly.express as px
import duckdb



def submissions_timeline(table_name, sql_db, cluster_num):
    df = sql_db.sql(f"SELECT created_uct FROM {table_name} Where cluster = {cluster_num}").fetchdf()

    # Convert the 'timestamp' column to datetime
    df['created_uct'] = pd.to_datetime(df['created_uct'])
    df = df.sort_values(by='created_uct')

    # Calculate the cumulative count of submissions
    df['cumulative_submissions'] = df.index + 1

    # Create the plot using Plotly
    fig = px.line(df, x='created_uct', y='cumulative_submissions', title='Total Number of Submissions Over Time')

    # Show the plot
    fig.show()


def subreddit_timeline(table_name, sql_db, cluster_num):
        df = sql_db.sql("SELECT created_uct, subreddit FROM {table_name} Where cluster = {cluster_num}").fetchdf()
        df['created_uct'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(by='created_uct')

        # Calculate the cumulative number of unique subreddits over time
        unique_subreddits = set()
        cumulative_counts = []

        for _, row in df.iterrows():
            unique_subreddits.add(row['subreddit'])
            cumulative_counts.append(len(unique_subreddits))

        df['cumulative_unique_subreddits'] = cumulative_counts

        fig = px.line(df, x='created_uct', y='cumulative_unique_subreddits',
                      title='Cumulative Number of Unique Subreddits Over Time')

        # Show the plot
        fig.show()


def author_timeline(table_name, sql_db, cluster_num):
    df = sql_db.sql("SELECT created_uct, author FROM {table_name} Where cluster = {cluster_num}").fetchdf()
    df['created_uct'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='created_uct')

    # Calculate the cumulative number of unique subreddits over time
    unique_subreddits = set()
    cumulative_counts = []

    for _, row in df.iterrows():
        unique_subreddits.add(row['author'])
        cumulative_counts.append(len(unique_subreddits))

    df['cumulative_unique_authors'] = cumulative_counts

    fig = px.line(df, x='created_uct', y='cumulative_unique_authors',
                  title='Cumulative Number of Unique Authors Over Time')

    # Show the plot
    fig.show()

def top_subreddit_timeline(table_name, sql_db, cluster_num):
    df = sql_db.sql("SELECT created_uct, subreddit FROM {table_name} Where cluster = {cluster_num}").fetchdf()
    df['created_uct'] = pd.to_datetime(df['created_uct'])

    # Extract the date from the timestamp
    df['date'] = df['created_uct'].dt.date

    # Group by date and get the top subreddit for each day
    top_subreddits = df.groupby('date')['subreddit'].agg(lambda x: x.mode()[0])

    # Reset the index to make it suitable for plotting
    top_subreddits = top_subreddits.reset_index()

    # Create the plot using Plotly
    fig = px.bar(top_subreddits, x='date', y='subreddit', title='Top Subreddit of Each Day',
                 labels={'subreddit': 'Top Subreddit'})

    # Show the plot
    fig.show()


