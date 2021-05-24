import numpy as np
from sklearn.utils import shuffle


def run(array):
    return {"array": shuffle(np.array(array)).tolist()}
