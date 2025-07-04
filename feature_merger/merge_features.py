import pandas as pd
import os

def merge_all_features(stock_df, crypto_feat_df, reddit_df, youtube_sentiment):
    # 1) Stock features DataFrame must already include:
    #    returns, sma50, sma200, rsi, bb_width
    stock_df['Date'] = pd.to_datetime(stock_df['Date']).dt.date

    # 2) Crypto features DataFrame must include 'Date' and price/crypto features
    crypto_feat_df['Date'] = pd.to_datetime(crypto_feat_df['Date']).dt.date

    # 3) Reddit daily sentiment
    if 'timestamp' in reddit_df.columns:
        reddit_df['Date'] = pd.to_datetime(reddit_df['timestamp'], unit='s').dt.date
        reddit_daily = reddit_df.groupby('Date', as_index=False)['sentiment'].mean()
    else:
        reddit_daily = pd.DataFrame({
            'Date': stock_df['Date'].unique(),
            'sentiment': 0.0
        })

    # 4) YouTube sentiment: accept either a DataFrame or a float
    if isinstance(youtube_sentiment, pd.DataFrame):
        youtube_sentiment['Date'] = pd.to_datetime(youtube_sentiment['Date']).dt.date
        yt_df = youtube_sentiment[['Date', 'youtube_sentiment']]
    else:
        # Broadcast a single float value across all dates
        yt_df = pd.DataFrame({
            'Date': stock_df['Date'],
            'youtube_sentiment': youtube_sentiment
        })

    # 5) Merge chain
    merged = (
        stock_df
        .merge(crypto_feat_df, on='Date', how='left')
        .merge(reddit_daily,    on='Date', how='left')
        .merge(yt_df,           on='Date', how='left')
    )

    # 6) Fill missing sentiment columns
    merged['sentiment'].fillna(0.0, inplace=True)
    merged['youtube_sentiment'].fillna(0.0, inplace=True)

    # 7) Save
    os.makedirs("data/merged", exist_ok=True)
    merged.to_csv("data/merged/merged_features.csv", index=False)
    print("Merged features saved to data/merged/merged_features.csv")
    return merged

if __name__ == "__main__":
    import ingestion.stock_data as S
    import ingestion.crypto_data_coingecko as C
    import osint.reddit_data   as R
    import osint.youtube_data  as Y

    stock   = S.download_stock("AAPL", start="2020-01-01", end="2023-12-31", interval="1d")
    crypto  = C.get_historical_data("bitcoin", days="90")
    reddit  = R.get_reddit_posts("wallstreetbets", limit=20)
    youtube = Y.get_youtube_transcript("dQw4w9WgXcQ")

    merge_all_features(stock, crypto, reddit, youtube)
