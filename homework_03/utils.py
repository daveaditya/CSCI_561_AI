import numpy as np

def read_data(file_path: str):
    return np.loadtxt(file_path, delimiter=",", dtype=np.float)

def read_labels(file_path: str):
    return np.loadtxt(file_path, delimiter=",", dtype=np.int32)

def write_predictions(file_path: str, predictions):
    np.savetxt(file_path, predictions, delimiter=",", fmt="%d", encoding="utf-8")