import matplotlib.pyplot as plt

from utils.utils import load_csv

def plot_scatter(dataset_path):
    """
    Generates a scatter plot for 'Defense Against the Dark Arts' vs. 'Astronomy',
    with points colored by Hogwarts House.
    """
    header, data = load_csv(dataset_path)
    
    try:
        idx1 = header.index('Astronomy')
        feature1_name = 'Astronomy'
        idx2 = header.index('Defense Against the Dark Arts')
        feature2_name = 'Defense Against the Dark Arts'
        house_idx = header.index('Hogwarts House')
    except ValueError:
        print("Error: Required columns not found in the dataset.")
        return

    houses = ['Gryffindor', 'Slytherin', 'Hufflepuff', 'Ravenclaw']
    colors = {'Gryffindor': 'red', 'Slytherin': 'green', 'Hufflepuff': 'yellow', 'Ravenclaw': 'blue'}
    plot_data = {house: {'x': [], 'y': []} for house in houses}

    for row in data:
        house = row[house_idx]
        if house in houses:
            try:
                val1 = float(row[idx1])
                val2 = float(row[idx2])
                plot_data[house]['x'].append(val1)
                plot_data[house]['y'].append(val2)
            except (ValueError, TypeError):
                continue

    plt.figure(figsize=(10, 8))
    
    for house, color in colors.items():
        plt.scatter(plot_data[house]['x'], plot_data[house]['y'], alpha=0.4, color=color, label=house)

    plt.title(f'Scatter Plot: {feature1_name} vs. {feature2_name}')
    plt.xlabel(feature1_name)
    plt.ylabel(feature2_name)
    plt.grid(True)
    plt.legend(title='Hogwarts House', loc='upper right')
    plt.show()