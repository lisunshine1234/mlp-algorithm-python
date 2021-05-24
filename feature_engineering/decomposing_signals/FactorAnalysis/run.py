import numpy as np
from sklearn.decomposition import FactorAnalysis


def run(array, label_index, n_components, tol, copy, max_iter, noise_variance_init, svd_method, iterated_power,
        random_state):
    x = np.array(array)

    if label_index is not None:
        if label_index < 0:
            label_index = len(array[0]) + label_index

        y = x[:, label_index]
        x = np.delete(array, label_index, axis=1)

    decomposition = FactorAnalysis(n_components=n_components,
                                   tol=tol,
                                   copy=copy,
                                   max_iter=max_iter,
                                   noise_variance_init=noise_variance_init,
                                   svd_method=svd_method,
                                   iterated_power=iterated_power,
                                   random_state=random_state)
    x = decomposition.fit_transform(x)
    if label_index is not None:
        x = np.insert(x, len(x[0]), values=y, axis=1)

    return {"array": x.tolist(),
            'components_': decomposition.components_.tolist(),
            'loglike_': decomposition.loglike_,
            'noise_variance_': decomposition.noise_variance_.tolist(),
            'n_iter_': decomposition.n_iter_,
            'mean_': decomposition.mean_.tolist()}
