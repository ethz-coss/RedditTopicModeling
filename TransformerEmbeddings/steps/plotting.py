import pandas as pd
import plotly.express as px
import steps.topic_finding as tf
import numpy as np
import statsmodels.api as sm

import duckdb


def size_comparison(table_name, sql_db):
    tfidf, terms = tf.get_tfidf(table_name, sql_db)
    top_words = tf.get_top_terms(tfidf, terms)

    # full 2 months data
    df = sql_db.sql(f"SELECT cluster, Count(*) as total FROM {table_name} "
                    f"Where cluster != -1 GROUP BY cluster ORDER BY total DESC").fetchdf()

    df['Graph'] = range(1, len(df) + 1)

    #do log. regression and find alpha
    log_sizes = np.log(df['total'])
    log_ranks = np.log(df['Graph'])

    X = sm.add_constant(log_sizes)

    model = sm.OLS(log_ranks, X)
    results = model.fit()

    alpha = results.params[1]
    beta = results.params[0]

    # Print the results
    print(f"Alpha (slope): {alpha}")
    print(f"Beta (intercept): {beta}")
    print(results.summary())

    #plot ranking
    fig = px.line(df, x='Graph', y='total',
                  labels={'Graph': 'Rank', 'total': 'Size'},
                  markers=True,
                  title='2 Month Cluster Size Distribution',
                  hover_data={'cluster': True},
                  template='plotly_white')

    fig.update_layout(
        title_font_size=30,
        yaxis_type='log',
        xaxis_type='log',
        xaxis_title_font_size=30,
        yaxis_title_font_size=30,
        xaxis_tickfont_size=25,
        yaxis_tickfont_size=25,
        font=dict(
            size=25,
            color='black'
        )
    )

    fig.write_html(f"./sizes_plot_2M.html")

    #repeat plotting for smaller timeframe
    df = sql_db.sql(f"SELECT cluster, Count(*) as total FROM {table_name} "
                    f"Where cluster != -1 AND created_utc > 1590425826 AND created_utc < 1592006399 "
                    f"GROUP BY cluster ORDER BY total DESC").fetchdf()
    df['Graph'] = range(1, len(df) + 1)

    fig = px.line(df, x='Graph', y='total',
                  labels={'Graph': 'Rank', 'total': 'Size'},
                  markers=True,
                  title='Hype Phase Cluster Size Distribution',
                  hover_data={'cluster': True},
                  template='plotly_white')

    fig.update_layout(
        title_font_size=30,
        yaxis_type='log',
        xaxis_type='log',
        xaxis_title_font_size=30,
        yaxis_title_font_size=30,
        xaxis_tickfont_size=25,
        yaxis_tickfont_size=25,
        font=dict(
            size=25,
            color='black'
        )
    )

    fig.write_html(f"./sizes_plot_HP.html")

def check(table_name, sql_db, cluster_num):
    df = sql_db.sql(
        f"SELECT COUNT(*) AS total, author FROM {table_name} Where cluster = {cluster_num} GROUP BY author ORDER BY total DESC ").fetchdf()
    print(df)


def top_subreddits(table_name, sql_db, cluster_num):
    df = sql_db.sql(f"SELECT subreddit, author FROM {table_name} Where cluster = {cluster_num}").fetchdf()

    # Aggregate the data by subreddit
    subreddit_counts = df['subreddit'].value_counts()

    # Separate top 6 subreddits and others
    top = subreddit_counts.nlargest(40)
    others = subreddit_counts.iloc[40:].sum()

    # Prepare data for pie chart
    labels = list(top.index) + ['Other']
    values = list(top.values) + [others]

    # Create pie chart
    fig = px.pie(
        names=labels,
        values=values,
        title="Subreddit Distribution"
    )

    t = [f"{v}" for v in values]
    fig.update_traces(
        text=t[:15],
        textinfo='text',
        hovertemplate='%{label}: %{value} submissions<extra></extra>',
        textfont_size=25
    )

    fig.update_layout(
        title_font_size=25,
        font=dict(
            size=25,
            color='black'
        ),
        legend=dict(
            font=dict(
                size=30  # Increase the font size of the legend here
            )
        ),
        width=1300,  # Set the desired width of the pie chart
        height=1000  # Set the desired height of the pie chart
    )

    # Show the plot
    fig.write_html(f"./subreddit_distribution_{cluster_num}.html")


def submissions_per_authors(table_name, sql_db, cluster_num):
    df = sql_db.sql(
        f"SELECT subreddit, author FROM {table_name} Where cluster = {cluster_num} AND author != '[deleted]'").fetchdf()

    author_submission_counts = df['author'].value_counts().reset_index()
    author_submission_counts.columns = ['author', 'submission_count']

    # Step 2: Count the number of authors for each submission count
    submission_count_distribution = author_submission_counts['submission_count'].value_counts().reset_index()
    submission_count_distribution.columns = ['submission_count', 'author_count']

    # Step 3: Create the block graph using Plotly
    fig = px.bar(submission_count_distribution,
                 x='submission_count',
                 y='author_count',
                 labels={'submission_count': 'Number of Submissions', 'author_count': 'Number of Authors'},
                 title='Number of Authors vs. Number of Submissions',
                 template='plotly_white')

    # Update layout for adjusting the font sizes of labels
    fig.update_layout(
        title_font_size=30,
        yaxis_type='log',
        yaxis_tickvals=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000],
        yaxis_ticktext=['1', '2', '5', '10', '20', '50', '100', '200', '500', '1000', '2000'],
        xaxis_title_font_size=30,
        yaxis_title_font_size=30,
        xaxis_tickfont_size=30,
        yaxis_tickfont_size=30,
        font=dict(
            size=20,
            color='black'
        )
    )

    # Show the plot
    fig.write_html(f"./author_submissions_{cluster_num}.html")


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

    long_df = merged_df.melt(id_vars='date', value_vars=['submissions', 'authors', 'unique subreddits'],
                             var_name='Type', value_name='Count')
    print(long_df.columns)

    # Create the line plot using plotly.express
    fig = px.line(long_df, x='date', y='Count', color='Type',
                  labels={'date': 'Date', 'Count': 'Count'},
                  markers=True,
                  title="Timeline over George Floyd's death",
                  template='plotly_white')

    fig.add_shape(
        type='line',
        x0='2020-05-25',
        x1='2020-05-25',
        y0=0,
        y1=(max(entries_per_day['submissions']) + 50),
        line=dict(color='black', width=2)
    )

    fig.add_annotation(
        x='2020-05-25',
        y=(max(entries_per_day['submissions']) + 50),
        text="George Floyd death",
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-40,
        font=dict(
            size=25  # Increase the font size here
        )
    )

    # Update layout for adjusting the font sizes of labels
    fig.update_layout(
        title_font_size=30,
        xaxis_title_font_size=30,
        yaxis_title_font_size=30,
        xaxis_tickfont_size=25,
        yaxis_tickfont_size=25,
        legend_title=dict(font=dict(size=25), text=''),
        legend=dict(
            x=0,
            y=1,
            font=dict(
                size=30  # Increase the font size of the legend here
            ),
            traceorder='normal',
            bgcolor='rgba(0,0,0,0)'
        ),
        font=dict(
            size=25,
            color='black'
        )
    )

    # Show the plot
    fig.write_html(f"./unique_per_day_{cluster_num}.html")


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
    fig = px.line(df, x='Date', y='Count', labels={'Date': 'Date', 'Count': 'Count'}, title='Cluster Size Over Time',
                  template='plotly_white')

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
