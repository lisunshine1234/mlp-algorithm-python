import numpy as np
from sklearn.decomposition import DictionaryLearning


def run(array, label_index, n_components, alpha, max_iter, tol, fit_algorithm, transform_algorithm,
        transform_n_nonzero_coefs,
        transform_alpha, n_jobs, code_init, dict_init, verbose, split_sign, random_state, positive_code, positive_dict,
        transform_max_iter):
    x = np.array(array)
    if label_index is not None:
        if label_index < 0:
            label_index = len(array[0]) + label_index

        y = x[:, label_index]
        x = np.delete(array, label_index, axis=1)

    decomposition = DictionaryLearning(n_components=n_components,
                                       alpha=alpha,
                                       max_iter=max_iter,
                                       tol=tol,
                                       fit_algorithm=fit_algorithm,
                                       transform_algorithm=transform_algorithm,
                                       transform_n_nonzero_coefs=transform_n_nonzero_coefs,
                                       transform_alpha=transform_alpha,
                                       n_jobs=n_jobs,
                                       code_init=code_init,
                                       dict_init=dict_init,
                                       verbose=verbose,
                                       split_sign=split_sign,
                                       random_state=random_state,
                                       positive_code=positive_code,
                                       positive_dict=positive_dict,
                                       transform_max_iter=transform_max_iter)
    # print(x)
    x = decomposition.fit_transform(x.tolist())
    if label_index is not None:
        x = np.insert(x, len(x[0]), values=y, axis=1)
    return {"array": x.tolist(),
            'components_': decomposition.components_.tolist(),
            'error_': decomposition.error_.tolist(),
            'n_iter_': decomposition.n_iter_}
