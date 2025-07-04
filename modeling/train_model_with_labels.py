# modeling/train_model_with_labels.py

import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

# 1) Load the labeled file
df = pd.read_csv("data/merged/labeled_features.csv", parse_dates=['Date'])
print("Loaded rows:", len(df))

# 2) Match these exactly with what merge+impute produces
features = [
    'returns',    'sma50',    'sma200',    'rsi',    'bb_width',
    'price',      'returns_crypto', 'sma50_crypto', 'sma200_crypto', 'rsi_crypto',
    'sentiment',  'youtube_sentiment'
]

# 3) Drop rows missing any feature or the target
df = df.dropna(subset=features + ['target'])
print("After dropping NaNs:", len(df))
if df.empty:
    raise RuntimeError("No samples remain after dropping missing values.")

X = df[features]
y = df['target']

# 4) Balance classes
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
print("After SMOTE:", X_res.shape, y_res.value_counts().to_dict())

# 5) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, shuffle=True
)

# 6) Fit
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7) Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# 8) Save
os.makedirs("modeling", exist_ok=True)
joblib.dump(model, "modeling/final_trading_model.pkl")
print("Model saved to modeling/final_trading_model.pkl")
