import pandas as pd

def generate_labels(
    input_path="data/merged/imputed_features.csv",
    output_path="data/merged/labeled_features.csv",
    price_column="Close",
    threshold=0.02,
    future_window=5
):
    df = pd.read_csv(input_path, parse_dates=['Date'])
    n = len(df)
    if n < future_window + 1:
        print(f"Not enough rows ({n}) for future window of {future_window}.")
        return

    prices = df[price_column].values
    labels = []
    for i in range(n):
        if i + future_window >= n:
            labels.append(0)
        else:
            change = (prices[i + future_window] - prices[i]) / prices[i]
            labels.append(int(change >= threshold))

    df['target'] = labels
    df.to_csv(output_path, index=False)
    print(f"Labels generated and saved to {output_path}")

if __name__ == "__main__":
    generate_labels()
