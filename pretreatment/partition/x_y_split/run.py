import numpy as np


def run(array, label_index):
    array = np.array(array)
    y = array[:, label_index]
    x = np.delete(array, label_index, axis=1)
    return {"x": x.tolist(), "y": y.tolist()}
