from utils.utils import load_csv, _clean_column, _count, _mean, _std, _min, _max, _percentile, describe_best_house

def describe(dataset_path):
    """
    Reads a dataset from a CSV file, computes descriptive statistics for all
    numerical features, and prints them in a formatted table.
    """
    header, data = load_csv(dataset_path)
    
    house_col_idx = -1
    try:
        house_col_idx = header.index('Hogwarts House')
    except ValueError:
        pass

    numerical_indices = []
    for i, value in enumerate(data[0]):
        if i == house_col_idx or header[i] == 'Index':
            continue
        try:
            float(value)
            numerical_indices.append(i)
        except (ValueError, TypeError):
            continue

    best_worst_houses = describe_best_house(header, data, numerical_indices)
    numerical_features = [header[i] for i in numerical_indices]
    stats = {feature: {} for feature in numerical_features}

    for i, feature_name in zip(numerical_indices, numerical_features):
        column_data = [row[i] for row in data]
        clean_data = _clean_column(column_data)
        
        if not clean_data:
            stats[feature_name] = {
                'Count': 0, 'Mean': 'N/A', 'Std': 'N/A', 'Min': 'N/A',
                '25%': 'N/A', '50%': 'N/A', '75%': 'N/A', 'Max': 'N/A',
                'Best House': 'N/A', 'Worst House': 'N/A'
            }
            continue

        stats[feature_name]['Count'] = _count(clean_data)
        stats[feature_name]['Mean'] = _mean(clean_data)
        stats[feature_name]['Std'] = _std(clean_data)
        stats[feature_name]['Min'] = _min(clean_data)
        stats[feature_name]['25%'] = _percentile(clean_data, 25)
        stats[feature_name]['50%'] = _percentile(clean_data, 50)
        stats[feature_name]['75%'] = _percentile(clean_data, 75)
        stats[feature_name]['Max'] = _max(clean_data)

        if feature_name in best_worst_houses:
            stats[feature_name]['Best House'] = best_worst_houses[feature_name]['best']
            stats[feature_name]['Worst House'] = best_worst_houses[feature_name]['worst']
        else:
            stats[feature_name]['Best House'] = 'N/A'
            stats[feature_name]['Worst House'] = 'N/A'

    print_table(stats, numerical_features)

def print_table(stats, features):
    """
    Prints the statistics in a well-formatted, transposed table.

    Args:
        stats (dict): A dictionary containing the computed statistics.
        features (list): A list of the feature names to display.
    """
    stat_names = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max', 'Best House', 'Worst House']
    col_width = 12 

    max_feature_len = max(len(f) for f in features) if features else 10
    feature_col_width = max(max_feature_len, len("Feature")) + 2

    header_line = f"| {'Feature':<{feature_col_width}} |"
    for name in stat_names:
        header_line += f" {name:>{col_width}} |"
    print("-" * len(header_line))
    print(header_line)
    print("-" * len(header_line))

    for feature in features:
        row_line = f"| {feature:<{feature_col_width}} |"
        for stat_name in stat_names:
            value = stats[feature][stat_name]
            if isinstance(value, (float, int)):
                row_line += f" {value:{col_width}.4f} |"
            else:
                row_line += f" {str(value):>{col_width}} |"
        print(row_line)
    
    print("-" * len(header_line))
    input("Press Enter to continue...")