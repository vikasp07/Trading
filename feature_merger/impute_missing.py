import pandas as pd
import os

def impute_missing_values():
    df = pd.read_csv("data/merged/merged_features.csv", parse_dates=['Date'])
    if df.empty:
        print("No merged features to impute.")
        return df

    # forward‐fill then back‐fill
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # any remaining NaNs → zero
    df.fillna(0, inplace=True)

    os.makedirs("data/merged", exist_ok=True)
    df.to_csv("data/merged/imputed_features.csv", index=False)
    print("Imputation completed and saved to data/merged/imputed_features.csv")
    return df

if __name__ == "__main__":
    impute_missing_values()
