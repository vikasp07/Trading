import yfinance as yf
import os

def download_stock(symbol, start, end, interval='1d'):
    """
    Downloads OHLCV data for `symbol`, resets index to get Date column,
    ALWAYS writes a CSV, and then warns if empty.
    """
    print(f"Downloading stock data for {symbol} (interval={interval})")
    
    # 1) Fetch data
    data = yf.download(symbol, start=start, end=end, interval=interval)
    
    # 2) Reset index so Date becomes a column (even if empty, index will be empty)
    data.reset_index(inplace=True)
    
    # 3) Ensure output folder exists
    out_dir = "data/stocks"
    os.makedirs(out_dir, exist_ok=True)
    file_path = f"{out_dir}/{symbol}_{interval}.csv"
    
    # 4) Always write the CSV
    data.to_csv(file_path, index=False)
    
    # 5) Warn if no rows were actually fetched
    if data.empty:
        print(f"Warning: No data returned for {symbol}. Check symbol or date range.")
    else:
        print(f"Saved {len(data)} rows of stock data to {file_path}")
    
    # 6) Return the DataFrame for immediate use (empty or not)
    return data
