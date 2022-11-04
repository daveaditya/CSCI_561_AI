import argparse

from utils import *

################################################################################################
### Constants
################################################################################################
PREDICTIONS_FILE = "./submission/test_predictions.csv"


################################################################################################
### Main Program
################################################################################################
def main(train_data_file ,train_label_file, test_data_file, test_label_file = None):

    # Load data and labels
    X_train, y_train = read_data(train_data_file), read_labels(train_label_file)
    X_test = read_data(test_data_file)

    is_testing = False
    if test_label_file:
        is_testing = True
        y_test = read_labels(test_label_file)

    # Train

    # Predict
    y_pred = None

    # Write Predictions
    write_predictions(PREDICTIONS_FILE, y_pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = "Multi-Layer Perceptron",
        description = "Learn and Classify 2-dimensional data using MLP.",
        epilog = "Neural Networks"
    )
    parser.add_argument("train_data_file", type=str)
    parser.add_argument("train_label_file", type=str)
    parser.add_argument("test_data_file", type=str)
    parser.add_argument("--test_label_file", required=False, type=str)

    args = parser.parse_args()
    main(args.train_data_file, args.train_label_file, args.test_data_file, args.test_label_file)