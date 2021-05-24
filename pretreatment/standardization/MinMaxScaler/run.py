import numpy as np
from sklearn.preprocessing import MinMaxScaler


def run(array, label_index, feature_range, copy):
    x = np.array(array)
    if label_index is not None:
        if label_index < 0:
            label_index = len(array[0]) + label_index

        y = x[:, label_index]
        x = np.delete(array, label_index, axis=1)

    scaler = MinMaxScaler(copy=copy, feature_range=feature_range)
    x = scaler.fit_transform(x)
    if label_index is not None:
        x = np.insert(x, label_index, values=y, axis=1)
    return {"array": x.tolist(),
            'min_': scaler.min_.tolist(),
            'scale_': scaler.scale_.tolist(),
            'data_min_': scaler.data_min_.tolist(),
            'data_max_': scaler.data_max_.tolist(),
            'data_range_': scaler.data_range_.tolist(),
            'n_samples_seen_': scaler.n_samples_seen_}
