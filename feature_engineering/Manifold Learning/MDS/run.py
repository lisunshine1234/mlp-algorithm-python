from sklearn.manifold import MDS

import numpy as np

def run(array,label_index, n_components, metric, n_init, max_iter, verbose, eps, n_jobs, random_state, dissimilarity
        ):
    x = np.array(array)
    if label_index is not None:
        if label_index < 0:
            label_index = len(array[0]) + label_index

        y = x[:, label_index]
        x = np.delete(array, label_index, axis=1)
    reg = MDS(n_components=n_components,
              metric=metric,
              n_init=n_init,
              max_iter=max_iter,
              verbose=verbose,
              eps=eps,
              n_jobs=n_jobs,
              random_state=random_state,
              dissimilarity=dissimilarity)

    x = reg.fit_transform(x).tolist()
    if label_index is not None:
        x = np.insert(x, len(x[0]), values=y, axis=1).tolist()
    return {'array': x,
            'embedding_': reg.embedding_.tolist(),
            'stress_': reg.stress_
            }
