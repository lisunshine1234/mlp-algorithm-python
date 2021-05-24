import numpy as np
from sklearn.preprocessing import PowerTransformer


def run(array, label_index,
        method,
        standardize,
        copy):
    x = np.array(array)

    if label_index is not None:
        if label_index < 0:
            label_index = len(array[0]) + label_index

        y = x[:, label_index]
        x = np.delete(array, label_index, axis=1)

    scaler = PowerTransformer(method=method, standardize=standardize, copy=copy)
    x = scaler.fit_transform(x)
    if label_index is not None:
        x = np.insert(x, label_index, values=y, axis=1)
    return {"array": x.tolist(), 'lambdas_': scaler.lambdas_.tolist()}
