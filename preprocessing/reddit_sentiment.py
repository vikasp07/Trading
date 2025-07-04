import pandas as pd
from textblob import TextBlob

def process_reddit_data(file_path):
    df = pd.read_csv(file_path)

    # Make sure 'created' column exists (or 'created_utc')
    if 'created_utc' in df.columns:
        df['timestamp'] = pd.to_datetime(df['created_utc'], unit='s')
    elif 'created' in df.columns:
        df['timestamp'] = pd.to_datetime(df['created'], unit='s')
    else:
        raise ValueError("No 'created_utc' or 'created' column found in reddit data.")

    # Calculate sentiment on titles (or full text if you want)
    df['sentiment'] = df['title'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

    return df[['timestamp', 'sentiment']]
            