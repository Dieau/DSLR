import math
import csv
from array import array

import numpy as np


def load_csv(path):
    """
    Loads a CSV file and returns its header and data.

    Args:
        path (str): The path to the CSV file.

    Returns:
        tuple: A tuple containing the list of headers (str) and a list of
               data rows (list of str).
    """
    try:
        with open(path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            data = list(reader)
        return header, data
    except FileNotFoundError:
        print(f"Error: File not found at {path}")
        exit(1)
    except Exception as e:
        print(f"An error occurred while reading the CSV: {e}")
        exit(1)


def preprocess_data(df, features, train_stats=None):
    """
    Prepares data for the model:
    1. Fills NaN values with the column mean.
    2. Standardizes features using Z-score normalization.
    3. Adds a bias term (intercept).
    
    Args:
        df (pd.DataFrame): The input dataframe.
        features (list): List of feature names to use.
        train_stats (dict, optional): Dictionary with mean and std from training data.
                                      If None, calculates them from the current df.
                                      
    Returns:
        tuple: (preprocessed_X_numpy, stats_dict)
    """
    X = df[features].copy()

    if train_stats is None:
        stats = {
            'mean': X.mean().to_dict(),
            'std': X.std().to_dict()
        }
    else:
        stats = train_stats

    X.fillna(stats['mean'], inplace=True)

    for col in features:
        mean = stats['mean'][col]
        std = stats['std'][col]
        if std != 0:
            X[col] = (X[col] - mean) / std
        else:
            X[col] = 0

    X.insert(0, 'Bias', 1)

    return X.to_numpy(), stats


def _is_nan(val):
    """Checks if a value is NaN."""
    return val != val


def _clean_column(column_data):
    """
    Converts a list of string values to floats and removes NaNs.

    Args:
        column_data (list): A list of values for a single feature.

    Returns:
        list: A list of floats with NaN values removed.
    """
    clean_data = []
    for item in column_data:
        try:
            val = float(item)
            if not _is_nan(val):
                clean_data.append(val)
        except (ValueError, TypeError):
            continue
    return clean_data


def _count(data):
    """
    Calculates the number of non-empty, valid numerical values in a list.

    Args:
        data (list): A list of numerical data.

    Returns:
        int: The count of valid numbers.
    """
    return len(data)


def _sum(data):
    """
    Calculates the sum of a list of numbers.

    Args:
        data (list): A list of numerical data.

    Returns:
        float: The sum of the numbers.
    """
    total = 0.0
    for x in data:
        total += x
    return total


def _mean(data):
    """
    Calculates the arithmetic mean of a list of numbers.

    Args:
        data (list): A list of numerical data.

    Returns:
        float: The mean of the data, or 0.0 if the list is empty.
    """
    if not data:
        return 0.0
    return _sum(data) / _count(data)


def _sqrt(n):
    if n < 0:
        return float('nan')
    if n == 0:
        return 0.0
    x = n
    y = (x + 1) / 2
    while y < x:
        x = y
        y = (x + n / x) / 2
    return x


def _std(data):
    """
    Calculates the sample standard deviation of a list of numbers.
    Args:
        data (list): A list of numerical data.
    Returns:
        float: The standard deviation, or 0.0 if there's not enough data.
    """
    n = _count(data)
    if n < 2:
        return 0.0
    mean_val = _mean(data)
    variance = _sum([(x - mean_val) ** 2 for x in data]) / (n - 1)
    return _sqrt(variance)


def _min(data):
    if not data:
        return None
    min_val = data[0]
    for x in data[1:]:
        if x < min_val:
            min_val = x
    return min_val


def _max(data):
    """
    Finds the maximum value in a list of numbers.

    Args:
        data (list): A list of numerical data.

    Returns:
        float: The maximum value, or None if the list is empty.
    """
    if not data:
        return None
    max_val = data[0]
    for x in data[1:]:
        if x > max_val:
            max_val = x
    return max_val


def _percentile(data, p):
    """
    Calculates the p-th percentile of a list of numbers.

    Args:
        data (list): A list of numerical data.
        p (int): The percentile to calculate (0-100).

    Returns:
        float: The value at the p-th percentile, or None if data is empty.
    """
    if not data:
        return None

    sorted_data = sorted(data)
    n = len(sorted_data)

    k = (p / 100) * (n + 1)

    if k < 1:
        return sorted_data[0]
    if k >= n:
        return sorted_data[-1]

    f = math.floor(k)
    c = math.ceil(k)

    if f == c:
        return sorted_data[int(k) - 1]

    d0 = sorted_data[int(f) - 1]
    d1 = sorted_data[int(c) - 1]
    return d0 + (d1 - d0) * (k - f)


def _clip(z, _min, _max):
    """
        Clip value
        Args:
            z: array of values
            _min: min value
            _max: max value

        Returns:
            list: clipped value(s)
    """
    if isinstance(z, np.ndarray):
        # Use NumPy's highly optimized C-based routines for performance
        return np.maximum(_min, np.minimum(z, _max))
    elif isinstance(z, list):
        return [max(_min, min(val, _max)) for val in z]
    else:
        return max(_min, min(z, _max))



def _exp(value):
    """
        Calculate the exponential of all elements in the input array.
    """
    if isinstance(value, list):
        return [_exp(x) for x in value]

    term = 1.0
    result = 1.0
    for i in range(1, 100):
        term *= value / i
        result += term
    return result

def describe_best_house(header, data, numerical_indices):
    """
    Calculates the best and worst performing house for each numerical feature.
    Args:
        header (list): The list of column headers.
        data (list): The dataset as a list of rows.
        numerical_indices (list): The indices of the numerical columns to analyze.
    Returns:
        dict: A dictionary where keys are feature names and values are dicts
              containing the 'best' and 'worst' house.
              e.g., {'Arithmancy': {'best': 'Ravenclaw', 'worst': 'Hufflepuff'}}
    """
    try:
        house_col_idx = header.index('Hogwarts House')
    except ValueError:
        return {}

    houses = ['Gryffindor', 'Slytherin', 'Hufflepuff', 'Ravenclaw']
    feature_names = [header[i] for i in numerical_indices]
    house_scores = {house: {name: [] for name in feature_names} for house in houses}

    for row in data:
        house = row[house_col_idx]
        if house in houses:
            for i, feature_name in zip(numerical_indices, feature_names):
                try:
                    score = float(row[i])
                    house_scores[house][feature_name].append(score)
                except (ValueError, TypeError):
                    continue

    house_means = {house: {feature: _mean(scores) for feature, scores in courses.items()}
                   for house, courses in house_scores.items()}

    results = {name: {'best': 'N/A', 'worst': 'N/A'} for name in feature_names}

    for feature_name in feature_names:
        feature_means_by_house = []
        for house in houses:
            mean_val = house_means[house].get(feature_name)
            if mean_val is not None and house_scores[house][feature_name]:
                feature_means_by_house.append((house, mean_val))

        if feature_means_by_house:
            best_house = max(feature_means_by_house, key=lambda item: item[1])[0]
            worst_house = min(feature_means_by_house, key=lambda item: item[1])[0]
            results[feature_name]['best'] = best_house
            results[feature_name]['worst'] = worst_house

    return results