# preprocessing/crypto_features.py

import pandas as pd

def process_crypto_data(file_path):
    """
    Reads a CSV with columns ['timestamp','price'], where timestamp
    may be either a millisecond Unix epoch or a datetime string.
    Produces rolling crypto features:
      • returns_crypto
      • sma50_crypto, sma200_crypto
      • rsi_crypto
    """
    df = pd.read_csv(file_path)

    # coerce price numeric
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df = df.dropna(subset=['price']).reset_index(drop=True)

    # parse timestamp into a DateTime
    ts = df['timestamp']
    if pd.api.types.is_numeric_dtype(ts):
        # numeric → epoch ms
        df['Date'] = pd.to_datetime(ts, unit='ms')
    else:
        # string → let pandas infer
        df['Date'] = pd.to_datetime(ts, errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

    # feature 1: returns
    df['returns_crypto'] = df['price'].pct_change()

    # feature 2: SMAs
    df['sma50_crypto']  = df['price'].rolling(window=50,  min_periods=1).mean()
    df['sma200_crypto'] = df['price'].rolling(window=200, min_periods=1).mean()

    # feature 3: RSI
    df['rsi_crypto'] = compute_rsi(df['price'])

    # drop any NaNs
    df = df.dropna().reset_index(drop=True)

    # keep only the columns merger needs
    return df[['Date','price','returns_crypto','sma50_crypto','sma200_crypto','rsi_crypto']]

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()

    rs  = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0)
