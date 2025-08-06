import argparse   
import numpy as np
import pandas as pd
import json
import os

from logreg_train import LogisticRegression
from utils.utils import preprocess_data

def predict(dataset_path, weights_path):
    if not os.path.exists(weights_path):
        print(f"Error: Weights file not found at {weights_path}. Please run logreg_train.py first.")
        return
    with open(weights_path, 'r') as f:
        model_data = json.load(f)

    weights = {house: np.array(w) for house, w in model_data['weights'].items()}
    features = model_data['features']
    train_stats = model_data['train_stats']
    houses = model_data['houses']

    try:
        df_test = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"Error: Test dataset not found at {dataset_path}")
        return

    if 'Careful_vs_Cunning' in features:
        df_test['Careful_vs_Cunning'] = df_test['Herbology'] - df_test['Ancient Runes']
    if 'Bravery_Score' in features:
        df_test['Bravery_Score'] = df_test['Flying'] - df_test['History of Magic']
    if 'Intellect_Score' in features:
        df_test['Intellect_Score'] = (df_test['Charms'] + df_test['Transfiguration']) - df_test['Muggle Studies']

    X_test, _ = preprocess_data(df_test, features, train_stats)

    all_probas = [LogisticRegression.sigmoid(np.dot(X_test, weights[house])) for house in houses]
    probas_matrix = np.stack(all_probas, axis=1)
    predictions_indices = np.argmax(probas_matrix, axis=1)
    predictions = [houses[i] for i in predictions_indices]

    submission_df = pd.DataFrame({
        'Index': df_test.index,
        'Hogwarts House': predictions
    })

    submission_df.to_csv('houses.csv', index=False)
    print("Prediction complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Hogwarts houses using a trained logistic regression model.")
    parser.add_argument("dataset", help="Path to the test dataset CSV file (e.g., dataset_test.csv).")
    parser.add_argument("weights", help="Path to the trained weights file (e.g., weights.json).")
    args = parser.parse_args()

    predict(args.dataset, args.weights)
