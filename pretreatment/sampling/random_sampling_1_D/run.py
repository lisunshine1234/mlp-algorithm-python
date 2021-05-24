import numpy as np


def run(array, size, replace, p):
    sample = np.random.choice(array, size, replace, p)
    return {"sample": sample.tolist()}
