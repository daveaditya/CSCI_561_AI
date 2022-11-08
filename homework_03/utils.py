import numpy as np
from math import floor


def read_data(file_path: str):
    return np.loadtxt(file_path, delimiter=",", dtype=np.float)


def read_labels(file_path: str):
    return np.loadtxt(file_path, delimiter=",", dtype=np.int32)


def write_predictions(file_path: str, predictions):
    np.savetxt(file_path, predictions, delimiter=",", fmt="%d", encoding="utf-8")


def train_test_split(X, y, test_ratio, rng):
    X_train, y_train, X_test, y_test = None, None, None, None

    n = X.shape[0]
    idxs = np.arange(n)
    rng.shuffle(idxs)

    split_idx = floor((1 - test_ratio) * n)
    train_idxs = idxs[:split_idx]
    test_idxs = idxs[split_idx:]

    X_train = X[train_idxs]
    y_train = y[train_idxs]
    X_test = X[test_idxs]
    y_test = y[test_idxs]

    return X_train, y_train, X_test, y_test
