import numpy as np
from sklearn.preprocessing import MaxAbsScaler


def run(array, label_index, copy):
    x = np.array(array)

    if label_index is not None:
        if label_index < 0:
            label_index = len(array[0]) + label_index

        y = x[:, label_index]
        x = np.delete(array, label_index, axis=1)

    scaler = MaxAbsScaler(copy=copy)
    x = scaler.fit_transform(x)
    if label_index is not None:
        x = np.insert(x, label_index, values=y, axis=1)
    return {"array": x.tolist(), "n_samples_seen_": scaler.n_samples_seen_,
            "max_abs_": scaler.max_abs_.tolist(), "scale_": scaler.scale_.tolist()}
