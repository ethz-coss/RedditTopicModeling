import pandas as pd
import plotly.express as px
import duckdb

def submissions_per_day(table_name, sql_db, cluster_num):
    df = sql_db.sql(f"SELECT created_utc FROM {table_name} Where cluster = {cluster_num}").fetchdf()

    df = df.sort_values(by='created_utc')
    # Convert the 'timestamp' column to datetime
    df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s')

    df['date'] = df['created_utc'].dt.date
    entries_per_day = df.groupby('date').size().reset_index(name='count')

    # Create a bar chart with Plotly
    fig = px.bar(entries_per_day, x='date', y='count', title='Entries per Day',
                 labels={'date': 'Date', 'count': 'Number of Entries'})

    # Show the plot
    fig.write_html(f"./submissions_per_day_{cluster_num}.html")


def submissions_timeline(table_name, sql_db, cluster_num):
    df = sql_db.sql(f"SELECT created_utc FROM {table_name} Where cluster = {cluster_num}").fetchdf()

    df = df.sort_values(by='created_utc')
    # Convert the 'timestamp' column to datetime
    df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s')
    # Calculate the cumulative count of submissions
    df['cumulative_submissions'] = range(1, len(df) + 1)
    # Create the plot using Plotly
    fig = px.line(df, x='created_utc', y='cumulative_submissions', title='Total Number of Submissions Over Time')

    # Show the plot
    fig.write_html(f"./submissions_timeline_{cluster_num}.html")

def subreddit_timeline(table_name, sql_db, cluster_num):
        df = sql_db.sql(f"SELECT created_utc, subreddit FROM {table_name} Where cluster = {cluster_num}").fetchdf()
        df = df.sort_values(by='created_utc')
        df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s')


        # Calculate the cumulative number of unique subreddits over time
        unique_subreddits = set()
        cumulative_counts = []

        for _, row in df.iterrows():
            unique_subreddits.add(row['subreddit'])
            cumulative_counts.append(len(unique_subreddits))

        df['cumulative_unique_subreddits'] = cumulative_counts

        fig = px.line(df, x='created_utc', y='cumulative_unique_subreddits',
                      title='Cumulative Number of Unique Subreddits Over Time')

        # Show the plot
        fig.write_html(f"./subreddit_timeline_{cluster_num}.html")


def author_timeline(table_name, sql_db, cluster_num):
    df = sql_db.sql(f"SELECT created_utc, author FROM {table_name} Where cluster = {cluster_num}").fetchdf()
    df = df.sort_values(by='created_utc')
    df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s')


    # Calculate the cumulative number of unique subreddits over time
    unique_subreddits = set()
    cumulative_counts = []

    for _, row in df.iterrows():
        unique_subreddits.add(row['author'])
        cumulative_counts.append(len(unique_subreddits))

    df['cumulative_unique_authors'] = cumulative_counts

    fig = px.line(df, x='created_utc', y='cumulative_unique_authors',
                  title='Cumulative Number of Unique Authors Over Time')

    # Show the plot
    fig.write_html(f"./unique_author_timeline_{cluster_num}.html")

def top_subreddit_timeline(table_name, sql_db, cluster_num):
    df = sql_db.sql(f"SELECT created_utc, subreddit FROM {table_name} Where cluster = {cluster_num}").fetchdf()
    df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s')

    # Extract the date from the timestamp
    df['date'] = df['created_utc'].dt.date

    # Group by date and get the top subreddit for each day
    top_subreddits = df.groupby('date')['subreddit'].agg(lambda x: x.mode()[0])

    # Reset the index to make it suitable for plotting
    top_subreddits = top_subreddits.reset_index()

    # Create the plot using Plotly
    fig = px.bar(top_subreddits, x='date', y='subreddit', title='Top Subreddit of Each Day',
                 labels={'subreddit': 'Top Subreddit'})

    # Show the plot
    fig.write_html("./top_subreddit_timeline.html")


