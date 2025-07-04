import joblib
import matplotlib.pyplot as plt

def plot_feature_importance(model_path, features):
    model = joblib.load(model_path)
    importances = model.feature_importances_

    sorted_idx = importances.argsort()[::-1]
    sorted_features = [features[i] for i in sorted_idx]
    sorted_importances = importances[sorted_idx]

    plt.figure(figsize=(10, 6))
    plt.barh(sorted_features, sorted_importances)
    plt.gca().invert_yaxis()
    plt.title("Feature Importance")
    plt.xlabel("Importance Score")
    plt.show()

# Run this
if __name__ == "__main__":
    features = [
        'returns_x', 'rsi_x', 'sma50_x', 'sma200_x', 'bb_width',
        'price', 'returns_y', 'rsi_y', 'sma50_y', 'sma200_y',
        'sentiment', 'youtube_sentiment'
    ]
    plot_feature_importance("modeling/tuned_trading_model.pkl", features)
