import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load data
df = pd.read_csv("data/merged/imputed_features.csv")

# Target label
df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df = df.dropna()

features = [
    'returns_x', 'rsi_x', 'sma50_x', 'sma200_x', 'bb_width',
    'price', 'returns_y', 'rsi_y', 'sma50_y', 'sma200_y',
    'sentiment', 'youtube_sentiment'
]

X = df[features]
y = df['target']

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42),
                           param_grid, cv=3, n_jobs=-1, verbose=1)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Evaluation
y_pred = best_model.predict(X_test)
print("Best Params:", grid_search.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save tuned model
joblib.dump(best_model, "modeling/tuned_trading_model.pkl")
print("Tuned model saved!")
