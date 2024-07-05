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
    df.rename(columns={'created_utc': 'date'}, inplace=True)

    # Create a bar chart with Plotly
    fig = px.line(entries_per_day, x='date', y='count', title='Entries per Day',
                 labels={'date': 'created_utc', 'count': 'Number of Entries'}, markers=True, template='plotly_white')


    fig.add_shape(
        type='line',
        x0='2020-05-25',
        x1='2020-05-25',
        y0=0,
        y1=max(entries_per_day['count']),
        line=dict(color='red', width=1)
    )

    fig.update_layout(
        title_font_size=24,
        xaxis_title_font_size=20,
        yaxis_title_font_size=20,
        xaxis_tickfont_size=14,
        yaxis_tickfont_size=14
    )

    # Show the plot
    fig.write_html(f"./submissions_per_day_{cluster_num}.html")

def unique_per_day(table_name, sql_db, cluster_num):

    df = sql_db.sql(f"SELECT created_utc, subreddit, author FROM {table_name} Where cluster = {cluster_num}").fetchdf()
    df = df.sort_values(by='created_utc')
    df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s')
    df['date'] = df['created_utc'].dt.date

    unique_subreddits_per_day = df.copy().groupby('date')['subreddit'].nunique().reset_index()
    unique_authors_per_day = df.copy().groupby('date')['author'].nunique().reset_index()
    entries_per_day = df.groupby('date').size().reset_index(name='submissions')

    merged_df = pd.merge(unique_authors_per_day, unique_subreddits_per_day, on='date')
    merged_df = pd.merge(merged_df, entries_per_day, on='date')
    merged_df.rename(columns={'submissions': 'submissions', 'author': 'authors', 'subreddit': 'unique subreddits'},
                     inplace=True)

    long_df = merged_df.melt(id_vars='date', value_vars=[ 'submissions', 'authors', 'unique subreddits'],
                             var_name='Type', value_name='Count')
    print(long_df.columns)

    # Create the line plot using plotly.express
    fig = px.line(long_df, x='date', y='Count', color='Type',
                  labels={'date': 'Date', 'Count': 'Count'},
                  markers=True,
                  title='Timeline over George Floyd death',
                  template='plotly_white')

    fig.add_shape(
        type='line',
        x0='2020-05-25',
        x1='2020-05-25',
        y0=0,
        y1=(max(entries_per_day['submissions'])+50),
        line=dict(color='black', width=1)
    )

    fig.add_annotation(
        x='2020-05-25',
        y=(max(entries_per_day['submissions'])+50),
        text="George Floyd death",
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-40
    )

    # Update layout for adjusting the font sizes of labels
    fig.update_layout(
        title_font_size=24,
        xaxis_title_font_size=20,
        yaxis_title_font_size=20,
        xaxis_tickfont_size=14,
        yaxis_tickfont_size=14,
        legend=dict(
            x=0,
            y=1,
            traceorder='normal',
            bgcolor='rgba(0,0,0,0)'
        ),
        font=dict(
            size=14,
            color='black'
        )
    )

    # Show the plot
    fig.write_html(f"./unique_per_day_{cluster_num}.html")


def authors_per_day(table_name, sql_db, cluster_num):

    df = sql_db.sql(f"SELECT created_utc, author FROM {table_name} Where cluster = {cluster_num}").fetchdf()
    df = df.sort_values(by='created_utc')
    df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s')
    df['date'] = df['created_utc'].dt.date
    unique_authors_per_day = df.groupby('date')['author'].nunique().reset_index()

    # Plotting the data using Plotly
    fig = px.line(unique_authors_per_day, x='date', y='author',
                 labels={'date': 'Date', 'author': 'Number of Unique Authors'},
                 markers=True,
                 title='Number of Unique Subreddits per Day',
                 template='plotly_white')

    # Update layout for adjusting the font sizes of labels
    fig.update_layout(
        title_font_size=24,
        xaxis_title_font_size=20,
        yaxis_title_font_size=20,
        xaxis_tickfont_size=14,
        yaxis_tickfont_size=14
    )

    # Show the plot
    fig.write_html(f"./authors_per_day_{cluster_num}.html")


def submissions_timeline(table_name, sql_db, cluster_num):
    df = sql_db.sql(f"SELECT created_utc FROM {table_name} Where cluster = {cluster_num}").fetchdf()

    df = df.sort_values(by='created_utc')
    # Convert the 'timestamp' column to datetime
    df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s')
    # Calculate the cumulative count of submissions
    df['cumulative_submissions'] = range(1, len(df) + 1)

    df.rename(columns={'created_utc': 'Date', 'cumulative_submissions': 'Count'},
              inplace=True)

    # Create the plot using Plotly
    fig = px.line(df, x='Date', y='Count', labels={'Date': 'Date', 'Count': 'Count'}, title='Cluster Size Over Time', template='plotly_white')


    fig.add_shape(
        type='line',
        x0='2020-05-25',
        x1='2020-05-25',
        y0=0,
        y1=max(df['Count']),
        line=dict(color='red', width=1)
    )

    fig.update_layout(
        title_font_size=24,
        xaxis_title_font_size=20,
        yaxis_title_font_size=20,
        xaxis_tickfont_size=14,
        yaxis_tickfont_size=14
    )
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
                      title='Cumulative Number of Unique Subreddits Over Time',
                      template='plotly_white')

        fig.update_layout(
            title_font_size=24,
            xaxis_title_font_size=20,
            yaxis_title_font_size=20,
            xaxis_tickfont_size=14,
            yaxis_tickfont_size=14
        )

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
                  title='Cumulative Number of Unique Authors Over Time',
                  template='plotly_white')

    fig.update_layout(
        title_font_size=24,
        xaxis_title_font_size=20,
        yaxis_title_font_size=20,
        xaxis_tickfont_size=14,
        yaxis_tickfont_size=14
    )

    # Show the plot
    fig.write_html(f"./unique_author_timeline_{cluster_num}.html")


