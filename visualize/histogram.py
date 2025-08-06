import matplotlib.pyplot as plt

from utils.utils import load_csv

def plot_histogram(dataset_path):
    """
    Generates histograms for each course to determine which has the most
    homogeneous score distribution across all four Hogwarts houses.

    :param dataset_path: Path to the dataset CSV file.
    """

    header, data = load_csv(dataset_path)
    
    houses = ['Gryffindor', 'Slytherin', 'Hufflepuff', 'Ravenclaw']
    colors = ['red', 'green', 'yellow', 'blue']
    
    course_indices = {header[i]: i for i in range(6, len(header))}
    
    house_scores = {house: {course: [] for course in course_indices} for house in houses}
    
    house_col_idx = header.index('Hogwarts House')
    for row in data:
        house = row[house_col_idx]
        if house in houses:
            for course, idx in course_indices.items():
                try:
                    score = float(row[idx])
                    house_scores[house][course].append(score)
                except (ValueError, TypeError):
                    continue

    fig, axes = plt.subplots(4, 4, figsize=(20, 15))
    fig.suptitle('Grade Distributions per Course for each Hogwarts House', fontsize=16)
    axes = axes.flatten()

    for i, course in enumerate(course_indices):
        ax = axes[i]
        for house, color in zip(houses, colors):
            ax.hist(house_scores[house][course], bins=20, alpha=0.5, label=house, color=color)
        ax.set_title(course)
        ax.set_xlabel('Grades')
        ax.set_ylabel('Students')

    for i in range(len(course_indices), len(axes)):
        fig.delaxes(axes[i])

    handles = [plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(len(houses))]
    fig.legend(handles, houses, loc='lower right', title='Hogwarts House')

    plt.subplots_adjust(hspace=1, wspace=0.5)
    plt.show()