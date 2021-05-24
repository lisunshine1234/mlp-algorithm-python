import numpy as np
from sklearn.decomposition import IncrementalPCA


def run(array, label_index, n_components, whiten, copy, batch_size):
    x = np.array(array)

    if label_index is not None:
        if label_index < 0:
            label_index = len(array[0]) + label_index

        y = x[:, label_index]
        x = np.delete(array, label_index, axis=1)

    decomposition = IncrementalPCA(n_components=n_components,
                                   whiten=whiten,
                                   copy=copy,
                                   batch_size=batch_size)
    x = decomposition.fit_transform(x)
    if label_index is not None:
        x = np.insert(x, len(x[0]), values=y, axis=1)

    return {"array": x.tolist(),
            'components_': decomposition.components_.tolist(),
            'explained_variance_': decomposition.explained_variance_.tolist(),
            'explained_variance_ratio_': decomposition.explained_variance_ratio_.tolist(),
            'singular_values_': decomposition.singular_values_.tolist(),
            'mean_': decomposition.mean_.tolist(),
            'var_': decomposition.var_.tolist(),
            'noise_variance_': decomposition.noise_variance_,
            'n_components_': decomposition.n_components_,
            'n_samples_seen_': int(decomposition.n_samples_seen_),
            'batch_size_': decomposition.batch_size_}
