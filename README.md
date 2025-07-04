# Trading Pipeline

A modular Python project to ingest market & OSINT data, engineer features, merge datasets, train a machine‐learning trading model, and run a paper‑trading simulation.

---

## 🚀 Features

- **Data Ingestion**  
  - Stock data via `yfinance`  
  - Crypto history via CoinGecko API  
  - Reddit sentiment from Pushshift/PRAW  
  - YouTube transcript sentiment  
- **Preprocessing & Feature Engineering**  
  - Stock: returns, SMA50, SMA200, RSI, Bollinger Bandwidth  
  - Crypto: price, returns, SMA, RSI  
  - Reddit & YouTube: daily average sentiment  
- **Feature Merger & Imputation**  
  - Join stock + crypto + OSINT on `Date`  
  - Handle missing values with mean‐imputation  
- **Label Generation**  
  - Binary “buy” signal if future price rises ≥ threshold in window  
- **Modeling**  
  - Random Forest classifier with SMOTE balancing  
  - Classification report & accuracy  
- **Simulation**  
  - PaperTradingSimulator uses model signals on historical `Close` prices  

---

