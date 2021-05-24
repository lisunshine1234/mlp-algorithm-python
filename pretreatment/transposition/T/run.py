import numpy as np


def run(array):
    array = np.array(array)
    return {"array": array.T.tolist()}
