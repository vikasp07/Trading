import pandas as pd

df = pd.read_csv("data/merged/labeled_features.csv", parse_dates=['Date'])

print("Checking missing values in labeled_features.csv...")
missing_summary = df.isnull().sum()
print("\nMissing values per column:\n", missing_summary)

print("\nTotal rows:", len(df))
print("Rows with ANY nulls:", df.isnull().any(axis=1).sum())

print("\nSample rows with nulls:")
print(df[df.isnull().any(axis=1)].head(10))
