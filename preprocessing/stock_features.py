import pandas as pd

def process_stock_data(file_path):
    df = pd.read_csv(file_path)
    # Parse and clean Date
    df = df.dropna(subset=['Date'])
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

    # Coerce numeric
    for col in ['Open','High','Low','Close','Adj Close','Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows without Close
    df = df.dropna(subset=['Close']).reset_index(drop=True)

    # 1) returns
    df['returns'] = df['Close'].pct_change()

    # 2) SMAs
    df['sma50']  = df['Close'].rolling(window=50,  min_periods=1).mean()
    df['sma200'] = df['Close'].rolling(window=200, min_periods=1).mean()

    # 3) RSI
    df['rsi'] = compute_rsi(df['Close'])

    # 4) Bollinger Bandwidth
    df['bb_width'] = compute_bollinger_bandwidth(df['Close'])

    # Drop any remaining NaNs
    df = df.dropna().reset_index(drop=True)
    return df

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()

    rs  = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0)

def compute_bollinger_bandwidth(series: pd.Series, period: int = 20) -> pd.Series:
    sma = series.rolling(window=period, min_periods=1).mean()
    std = series.rolling(window=period, min_periods=1).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    bandwidth = (upper - lower) / sma
    return bandwidth.fillna(0)
