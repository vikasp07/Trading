import finnhub
import pandas as pd
from config.config import FINNHUB_API_KEY
from utils.logger import log_info

finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

def get_realtime_quote(symbol):
    log_info(f"Fetching real-time quote for {symbol}")
    quote = finnhub_client.quote(symbol)
    return quote
