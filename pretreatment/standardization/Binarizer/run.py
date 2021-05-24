import numpy as np
from sklearn.preprocessing import Binarizer


def run(array, label_index, threshold, copy):
    x = np.array(array)

    if label_index is not None:
        if label_index < 0:
            label_index = len(array[0]) + label_index

        y = x[:, label_index]
        x = np.delete(array, label_index, axis=1)

    scaler = Binarizer(copy=copy, threshold=threshold).fit(x)
    x = scaler.transform(x)
    if label_index is not None:
        x = np.insert(x, label_index, values=y, axis=1)
    return {"array": x.tolist()}
