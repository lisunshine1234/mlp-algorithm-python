import numpy as np
from sklearn.preprocessing import StandardScaler


def run(array, label_index, copy, with_mean, with_std):
    x = np.array(array)

    if label_index is not None:
        if label_index < 0:
            label_index = len(array[0]) + label_index

        y = x[:, label_index]
        x = np.delete(array, label_index, axis=1)

    scaler = StandardScaler(copy=copy, with_mean=with_mean, with_std=with_std).fit(x)

    x = scaler.transform(x)
    if label_index is not None:
        x = np.insert(x, label_index, values=y, axis=1)
    return {"array": x.tolist()}
