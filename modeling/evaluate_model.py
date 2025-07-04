import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Define features (same as training)
FEATURES = [
    'returns_x', 'rsi_x', 'sma50_x', 'sma200_x', 'bb_width',
    'price', 'returns_y', 'rsi_y', 'sma50_y', 'sma200_y',
    'sentiment', 'youtube_sentiment'
]

def evaluate_model():
    # Load model
    model = joblib.load("modeling/tuned_trading_model.pkl")

    # Load dataset
    data = pd.read_csv("data/merged/imputed_features.csv")

    # For demo purpose: create a dummy target column (in real use-case this should be true labels)
    # Let's simulate some target using random 0/1 (since we don’t have actual labels)
    # REMOVE this part if you have real target data
    import numpy as np
    if 'target' not in data.columns:
        data['target'] = np.random.randint(0, 2, size=len(data))

    X = data[FEATURES]
    y_true = data['target']

    # Predict
    y_pred = model.predict(X)

    # Evaluation Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print("\nModel Evaluation Results:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

if __name__ == "__main__":
    evaluate_model()
