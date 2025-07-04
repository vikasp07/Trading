# main.py

import pandas as pd

from ingestion.stock_data             import download_stock
from preprocessing.stock_features     import process_stock_data

from ingestion.crypto_data_coingecko  import get_historical_data
from preprocessing.crypto_features    import process_crypto_data

from osint.reddit_data                import get_reddit_posts
from osint.youtube_data               import get_youtube_transcript

from feature_merger.merge_features    import merge_all_features
from feature_merger.impute_missing    import impute_missing_values

if __name__ == "__main__":
    # Phase 1: Ingest raw OHLCV
    download_stock("AAPL",   start="2020-01-01", end="2023-12-31", interval="1d")
    get_historical_data("bitcoin", days="90")
    get_historical_data("ethereum", days="90")
    get_reddit_posts("wallstreetbets", limit=20)
    # This returns a single float or DataFrame; see merge_features
    yt = get_youtube_transcript("dQw4w9WgXcQ")

    # Phase 2: Feature‐engineer
    stock_df = process_stock_data("data/stocks/AAPL_1d.csv")
    btc_df   = process_crypto_data("data/crypto/bitcoin_90d.csv")
    eth_df   = process_crypto_data("data/crypto/ethereum_90d.csv")

    reddit_df = pd.read_csv("data/osint/reddit_wallstreetbets.csv")

    # Phase 3: Merge + Impute
    merged = merge_all_features(stock_df, btc_df, reddit_df, yt)
    imputed = impute_missing_values()

    print("Pipeline completed successfully!")
