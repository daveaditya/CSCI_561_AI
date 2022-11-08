import argparse

from utils import *
from mlp import *

################################################################################################
### Constants
################################################################################################
PREDICTIONS_FILE = "./submission/test_predictions.csv"

RANDOM_SEED = 42

np.random.seed(seed=RANDOM_SEED)
rng = np.random.default_rng(seed=RANDOM_SEED)
################################################################################################
### Main Program
################################################################################################
def main(train_data_file, train_label_file, test_data_file, test_label_file=None):

    # Load data and labels
    X, y = read_data(train_data_file), read_labels(train_label_file)
    X_test = read_data(test_data_file)

    # Add features
    X1 = X[:, 0]
    X2 = X[:, 1]
    X_new = np.stack([X1, X2, np.power(X1, 2), np.power(X2, 2), np.multiply(X1, X2), np.sin(X1), np.sin(X2)], axis=1)

    X1_test = X_test[:, 0]
    X2_test = X_test[:, 1]
    X_test_new = np.stack(
        [
            X1_test,
            X2_test,
            np.power(X1_test, 2),
            np.power(X2_test, 2),
            np.multiply(X1_test, X2_test),
            np.sin(X1_test),
            np.sin(X2_test),
        ],
        axis=1,
    )

    is_testing = False
    if test_label_file:
        is_testing = True
        y_test = read_labels(test_label_file)

    # Define Model
    model = dict()
    model["FC_1"] = LinearLayer(in_dim=7, out_dim=7, rng=rng)
    model["RELU_1"] = ReLU()
    model["FC_2"] = LinearLayer(in_dim=7, out_dim=7, rng=rng)
    model["RELU_2"] = ReLU()
    model["FC_3"] = LinearLayer(in_dim=7, out_dim=7, rng=rng)
    model["RELU_3"] = ReLU()
    model["FC_4"] = LinearLayer(in_dim=7, out_dim=2, rng=rng)
    model["LOSS"] = SoftmaxCrossEntropy()

    # Train
    model = train(
        X_new,
        y,
        val_ratio=0.2,
        model=model,
        n_epoch=100,
        mini_batch_size=8,
        alpha=0.01,
        learning_rate=0.03,
        step=50,
        rng=rng,
    )

    # Predict
    y_pred = predict(model, X_test_new)

    if is_testing:
        from sklearn.metrics import classification_report
        print(classification_report(y_test, y_pred))

    # Write Predictions
    write_predictions(PREDICTIONS_FILE, y_pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Multi-Layer Perceptron",
        description="Learn and Classify 2-dimensional data using MLP.",
        epilog="Neural Networks",
    )
    parser.add_argument("train_data_file", type=str)
    parser.add_argument("train_label_file", type=str)
    parser.add_argument("test_data_file", type=str)
    parser.add_argument("--test_label_file", required=False, type=str)

    args = parser.parse_args()
    main(args.train_data_file, args.train_label_file, args.test_data_file, args.test_label_file)
