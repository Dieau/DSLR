import argparse
import os

import logreg_predict
import train
from tools.accuracy_check import check_accuracy
from visualize.describe import describe
from visualize.histogram import plot_histogram
from visualize.pair_plot import plot_pair
from visualize.scatter_plot import plot_scatter

# --- Style Definitions ---
class Colors:
    """ANSI escape sequences for terminal colors and formatting"""
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    NC = '\033[0m'

dataset_train = None
dataset_test = None
dataset_truth = None

def clear_screen():
    """
    Clears the console screen.
    """
    os.system('cls' if os.name == 'nt' else 'clear')
    print(r"""
                                        ,--,               
                                     ,---.'|               
                ,---,      .--.--.   |   | :   ,-.----.    
              .'  .' `\   /  /    '. :   : |   \    /  \   
            ,---.'     \ |  :  /`. / |   ' :   ;   :    \  
            |   |  .`\  |;  |  |--`  ;   ; '   |   | .\ :  
            :   : |  '  ||  :  ;_    '   | |__ .   : |: |  
            |   ' '  ;  : \  \    `. |   | :.'||   |  \ :  
            '   | ;  .  |  `----.   \'   :    ;|   : .  /  
            |   | :  |  '  __ \  \  ||   |  ./ ;   | |  \  
            '   : | /  ;  /  /`--'  /;   : ;   |   | ;\  \ 
            |   | '` ,/  '--'.     / |   ,/    :   ' | \.' 
            ;   :  .'      `--'---'  '---'     :   : :-'   
            |   ,.'                            |   |.'     
            '---'                              `---'       
            """)

def start_app():
    while True:
        clear_screen()
        print("v - Visualize and describe dataset")
        print("t - Train model")
        print("p - Predict")
        print("q - Quit")

        choice = input("Enter your choice: ")

        clear_screen()
        if choice == 'v':
            visualize_menu()
        if choice == 't':
            train.main()
        elif choice == 'p':
            logreg_predict.predict(dataset_test, 'weights.json')
            check_accuracy('houses.csv', dataset_truth)
        elif choice == 'q':
            print("Exiting the application.")
            exit(0)
        else:
            print("Invalid choice. Please try again.")

        input("Press enter to continue...")

def visualize_menu():
    """
    Displays a menu for visualizing the dataset.
    """
    while True:
        clear_screen()
        print("Visualize and Describe Dataset")
        print("d - Describe dataset")
        print("h - Draw histograms")
        print("s - Draw scatter plots")
        print("p - Draw pair plots")
        print("m - Back to main menu")
        print("q - Quit")

        choice = input("Enter your choice: ")

        if choice == 'd':
            clear_screen()
            describe(dataset_train)
        elif choice == 'h':
            plot_histogram(dataset_train)
        elif choice == 's':
            plot_scatter(dataset_train)
        elif choice == 'p':
            plot_pair(dataset_train)
        elif choice == 'm':
            start_app()
        elif choice == 'q':
            print("Exiting the application.")
            exit(0)
        else:
            print("Invalid choice. Please try again.")
            input("Press enter to continue...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot histograms to find the most homogeneous course distribution.")
    parser.add_argument("dataset", help="Path to the dataset CSV file.")
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f"Error: Dataset file not found at '{args.dataset}'")
        exit(1)

    dataset_train = args.dataset
    dataset_test = args.dataset.replace('train', 'test')
    dataset_truth = args.dataset.replace('train', 'truth')

    start_app()