import matplotlib.pyplot as plt
import pandas as pd

def format_label(label):
    """
    Inserts a newline character after every word in the label for vertical alignment.
    """
    return '\n'.join(label.split())

def plot_pair(dataset_path):
    """
    Generates a pair plot (scatter plot matrix) for course scores,
    """
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"Error: File not found at {dataset_path}")
        return

    courses = [
        'Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts',
        'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic',
        'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms', 'Flying'
    ]
    df_plot = df[courses + ['Hogwarts House']].dropna()

    houses = df_plot['Hogwarts House'].unique()
    house_colors = {'Gryffindor': 'red', 'Slytherin': 'green', 'Hufflepuff': 'yellow', 'Ravenclaw': 'blue'}
    
    num_features = len(courses)
    fig, axes = plt.subplots(num_features, num_features, figsize=(25, 25))

    for i, feature_y in enumerate(courses):
        for j, feature_x in enumerate(courses):
            ax = axes[i, j]
            
            if j == 0:
                ax.set_ylabel(format_label(feature_y), fontsize=7)
            
            if i == num_features - 1:
                ax.set_xlabel(format_label(feature_x), fontsize=7)

            ax.set_xticks([])
            ax.set_yticks([])

            if i == j:
                for house in houses:
                    ax.hist(df_plot[df_plot['Hogwarts House'] == house][feature_x],
                            bins=15, alpha=0.5, color=house_colors[house], density=True)
            else:
                for house in houses:
                    ax.scatter(df_plot[df_plot['Hogwarts House'] == house][feature_x],
                               df_plot[df_plot['Hogwarts House'] == house][feature_y],
                               alpha=0.3, color=house_colors[house], s=5, label=house)

    handles = [plt.Line2D([0], [0], marker='o', color='w', label=h,
                          markerfacecolor=c, markersize=10) for h, c in house_colors.items()]
    fig.legend(handles=handles, title='Hogwarts House', loc='lower right')

    plt.subplots_adjust(left=0.08, bottom=0.08, right=0.92, top=0.97, wspace=0.05, hspace=0.05)
    plt.show()