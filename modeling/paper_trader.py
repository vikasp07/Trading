# modeling/paper_trader.py

import pandas as pd
import joblib

class PaperTradingSimulator:
    def __init__(self, model_path, data_path, features, initial_balance=10000):
        self.model = joblib.load(model_path)
        df = pd.read_csv(data_path, parse_dates=['Date'])
        # Drop any rows missing the features we'll use or the 'Close' price
        self.data = df.dropna(subset=features + ['Close'])
        self.features = features
        self.balance = initial_balance
        self.position = 0
        self.trade_log = []

    def simulate(self):
        for _, row in self.data.iterrows():
            # Build a 1‑row DataFrame so sklearn sees feature names
            X = pd.DataFrame([row[self.features]], columns=self.features)
            signal = self.model.predict(X)[0]

            price = row['Close']
            date_str = row['Date'].strftime("%Y-%m-%d")

            if signal == 1 and self.balance >= price:
                self.position += 1
                self.balance -= price
                self.trade_log.append(f"BUY  at ${price:.2f} on {date_str}")
            elif signal == 0 and self.position > 0:
                self.position -= 1
                self.balance += price
                self.trade_log.append(f"SELL at ${price:.2f} on {date_str}")

        # Liquidate at final close
        if self.position > 0:
            final_price = self.data.iloc[-1]['Close']
            self.balance += self.position * final_price
            self.trade_log.append(f"LIQUIDATE {self.position} at ${final_price:.2f}")
            self.position = 0

        # Print results
        print("Trade Log:")
        if not self.trade_log:
            print("none")
        else:
            for entry in self.trade_log:
                print(entry)
        print(f"\nFinal Balance: ${self.balance:.2f}")

if __name__ == "__main__":
    features = [
        'returns', 'sma50', 'sma200', 'rsi', 'bb_width',
        'price', 'returns_crypto', 'sma50_crypto', 'sma200_crypto', 'rsi_crypto',
        'sentiment', 'youtube_sentiment'
    ]

    sim = PaperTradingSimulator(
        model_path="modeling/final_trading_model.pkl",
        data_path="data/merged/labeled_features.csv",
        features=features,
        initial_balance=10000
    )
    sim.simulate()
