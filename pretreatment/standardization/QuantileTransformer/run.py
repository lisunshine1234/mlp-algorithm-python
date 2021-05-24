import numpy as np
from sklearn.preprocessing import QuantileTransformer


def run(array, label_index, n_quantiles, output_distribution, ignore_implicit_zeros, subsample, random_state, copy):
    x = np.array(array)

    if label_index is not None:
        if label_index < 0:
            label_index = len(array[0]) + label_index

        y = x[:, label_index]
        x = np.delete(array, label_index, axis=1)

    scaler = QuantileTransformer(n_quantiles=n_quantiles, output_distribution=output_distribution,
                                 ignore_implicit_zeros=ignore_implicit_zeros, subsample=subsample,
                                 random_state=random_state, copy=copy)
    x = scaler.fit_transform(x)
    if label_index is not None:
        x = np.insert(x, label_index, values=y, axis=1)
    return {"array": x.tolist(),
            "references_": scaler.references_.tolist(),
            "quantiles_": scaler.quantiles_.tolist(),
            "n_quantiles_": scaler.n_quantiles_}
