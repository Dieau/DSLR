import argparse
import numpy as np
import pandas as pd
import json

from utils.utils import preprocess_data, _clip, _exp


class LogisticRegression:
    def __init__(self, learning_rate, iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None

    @staticmethod
    def sigmoid(z):
        z = _clip(z, -500, 500)
        return 1 / (1 + _exp(-z))

    def fit(self, X, y):
        m_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        for _ in range(self.iterations):
            z = np.dot(X, self.weights)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / m_samples
            self.weights -= self.learning_rate * gradient

def create_engineered_features(df):
    """Creates and adds the engineered features to the dataframe."""
    df['Careful_vs_Cunning'] = df['Herbology'] - df['Ancient Runes']
    df['Bravery_Score'] = df['Flying'] - df['History of Magic']
    df['Intellect_Score'] = (df['Charms'] + df['Transfiguration']) - df['Muggle Studies']
    return df

def train(dataset_path, learning_rate, iterations):
    """
    Trains the model on the given dataset and saves the final weights.
    """
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"Error: File not found at {dataset_path}")
        return

    print("Creating engineered features...")
    df = create_engineered_features(df)
    
    features = [
        'Defense Against the Dark Arts', 'Divination', 'Muggle Studies',
        'Flying', 'Careful_vs_Cunning', 'Bravery_Score', 'Intellect_Score'
    ]
    
    X, train_stats = preprocess_data(df, features)
    
    houses = sorted(df['Hogwarts House'].unique())
    all_weights = {}

    print("Training models for each house (One-vs-All)...")
    for house in houses:
        print(f"  - Training for {house}...")
        y = (df['Hogwarts House'] == house).astype(int).to_numpy()
        
        model = LogisticRegression(learning_rate=learning_rate, iterations=iterations)
        model.fit(X, y)
        
        all_weights[house] = model.weights.tolist()

    model_data = {
        'weights': all_weights,
        'features': features,
        'train_stats': train_stats,
        'houses': houses
    }
    
    with open('weights.json', 'w') as f:
        json.dump(model_data, f, indent=4)
        
    print("\nTraining complete. Weights saved to weights.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a logistic regression model.")
    parser.add_argument("dataset", help="Path to the training dataset CSV file.")
    parser.add_argument("--lr", type=float, default=0.1, help="The learning rate for gradient descent.")
    parser.add_argument("--iter", type=int, default=5000, help="The number of iterations for gradient descent.")
    args = parser.parse_args()
    
    train(args.dataset, args.lr, args.iter)
