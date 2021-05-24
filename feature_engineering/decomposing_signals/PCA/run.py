import numpy as np
from sklearn.decomposition import PCA


def run(array, label_index, n_components, copy, whiten, svd_solver, tol, iterated_power, random_state):
    x = np.array(array)
    if label_index is not None:
        if label_index < 0:
            label_index = len(array[0]) + label_index

        y = x[:, label_index]
        x = np.delete(array, label_index, axis=1)

    decomposition = PCA(n_components=n_components,
                        copy=copy,
                        whiten=whiten,
                        svd_solver=svd_solver,
                        tol=tol,
                        iterated_power=iterated_power,
                        random_state=random_state)
    x = decomposition.fit_transform(x)
    if label_index is not None:
        x = np.insert(x, len(x[0]), values=y, axis=1)

    return {"array": x.tolist(),
            'components_': decomposition.components_.tolist(),
            'explained_variance_': decomposition.explained_variance_.tolist(),
            'explained_variance_ratio_': decomposition.explained_variance_ratio_.tolist(),
            'singular_values_': decomposition.singular_values_.tolist(),
            'mean_': decomposition.mean_.tolist(),
            'n_components_': decomposition.n_components_,
            'n_features_': decomposition.n_features_,
            'n_samples_': decomposition.n_samples_,
            'noise_variance_': decomposition.noise_variance_}
