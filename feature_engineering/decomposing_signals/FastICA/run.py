import numpy as np
from sklearn.decomposition import FastICA


def run(array, label_index, n_components, algorithm, whiten, fun, fun_args, max_iter, tol, w_init, random_state):
    x = np.array(array)

    if label_index is not None:
        if label_index < 0:
            label_index = len(array[0]) + label_index

        y = x[:, label_index]
        x = np.delete(array, label_index, axis=1)

    decomposition = FastICA(n_components=n_components,
                            algorithm=algorithm,
                            whiten=whiten,
                            fun=fun,
                            fun_args=fun_args,
                            max_iter=max_iter,
                            tol=tol,
                            w_init=w_init,
                            random_state=random_state)
    x = decomposition.fit_transform(x)
    if label_index is not None:
        x = np.insert(x, len(x[0]), values=y, axis=1)

    return {"array": x.tolist(),
            'components_': decomposition.components_.tolist(),
            'mean_': decomposition.mean_.tolist(),
            'n_iter_': decomposition.n_iter_,
            'whitening_': decomposition.whitening_.tolist()}
