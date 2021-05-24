import numpy as np


def run(file_name, delimiter):
    array = np.loadtxt(file_name, delimiter=delimiter)
    return {"array": np.array(array, dtype=float).tolist()}
