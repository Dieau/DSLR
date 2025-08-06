import argparse
import pandas as pd
import sys

BLUE = '\033[94m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
CYAN = '\033[96m'
MAGENTA = '\033[95m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'
NC = '\033[0m'

def check_accuracy(predictions_path, truth_path):
    """
    Loads prediction and truth files, compares them, and prints the accuracy.
    """
    try:
        df_pred = pd.read_csv(predictions_path)
        df_true = pd.read_csv(truth_path)
    except FileNotFoundError as e:
        print(f"Accuracy check failed: {e}", file=sys.stderr)
        sys.exit(1)

    df_merged = pd.merge(df_pred, df_true, on='Index', suffixes=('_pred', '_true'), how="inner", validate="many_to_many")

    if len(df_merged) != len(df_true):
        print(
            f"Warning: Mismatch in record count. Truth has {len(df_true)} rows, but only {len(df_merged)} matched in predictions.",
            file=sys.stderr
        )

    correct = (df_merged['Hogwarts House_pred'] == df_merged['Hogwarts House_true']).sum()
    total = len(df_merged)
    accuracy = (correct / total) * 100 if total > 0 else 0

    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculates the accuracy of predictions against a truth file.")
    parser.add_argument("predictions_file", help="Path to the predictions CSV file (e.g., houses.csv).")
    parser.add_argument("truth_file", help="Path to the ground truth CSV file (e.g., dataset_truth.csv).")
    args = parser.parse_args()
    
    check_accuracy(args.predictions_file, args.truth_file)
