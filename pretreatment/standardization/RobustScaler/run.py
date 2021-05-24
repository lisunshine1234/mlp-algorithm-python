import numpy as np
from sklearn.preprocessing import RobustScaler


def run(array, label_index, with_centering, with_scaling, quantile_range, copy):
    x = np.array(array)

    if label_index is not None:
        if label_index < 0:
            label_index = len(array[0]) + label_index

        y = x[:, label_index]
        x = np.delete(array, label_index, axis=1)

    scaler = RobustScaler(with_centering=with_centering,
                          with_scaling=with_scaling,
                          quantile_range=quantile_range,
                          copy=copy).fit(x)
    x = scaler.transform(x)
    if label_index is not None:
        x = np.insert(x, label_index, values=y, axis=1)
    return {"array": x.tolist(), 'scale_': scaler.scale_.tolist(), 'center_': scaler.center_.tolist()}
