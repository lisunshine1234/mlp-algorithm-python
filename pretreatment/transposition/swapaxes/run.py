import numpy as np


def run(array, axis1, axis2):
    array = np.array(array)
    return {"array": array.swapaxes(axis1, axis2).tolist()}
