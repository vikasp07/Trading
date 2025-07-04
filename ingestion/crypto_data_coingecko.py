import requests
import pandas as pd
import time
from utils.logger import log_info, log_error

BASE_URL = "https://api.coingecko.com/api/v3"

def get_historical_data(coin_id, vs_currency='usd', days='90'):
    """
    Downloads hourly OHLC price data for the past `days` days (<=90)
    from CoinGecko and saves to CSV.
    """
    log_info(f"Downloading historical data for {coin_id}")
    url = f"{BASE_URL}/coins/{coin_id}/market_chart"
    params = {
        'vs_currency': vs_currency,
        'days': days
        # interval is omitted—free tier auto‑provides hourly for days ≤ 90
    }

    while True:
        resp = requests.get(url, params=params)
        if resp.status_code == 200:
            data = resp.json()
            if 'prices' not in data:
                log_error(f"No 'prices' in response for {coin_id}: {data}")
                return
            df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            out_path = f"data/crypto/{coin_id}_{days}d.csv"
            df.to_csv(out_path, index=False, encoding='utf-8')
            log_info(f"Saved {coin_id} data to {out_path}")
            break

        elif resp.status_code == 429:
            log_error("Rate limited by CoinGecko; sleeping 60s...")
            time.sleep(60)
        else:
            log_error(f"Error {resp.status_code} for {coin_id}: {resp.text}")
            return
