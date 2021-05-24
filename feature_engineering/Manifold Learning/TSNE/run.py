from sklearn.manifold import TSNE

import numpy as np


def run(array, label_index, n_components, perplexity, early_exaggeration, learning_rate, n_iter, n_iter_without_progress, min_grad_norm, metric, init, verbose,
        random_state,
        method, angle, n_jobs
        ):
    x = np.array(array)
    if label_index is not None:
        if label_index < 0:
            label_index = len(array[0]) + label_index

        y = x[:, label_index]
        x = np.delete(array, label_index, axis=1)
    reg = TSNE(n_components=n_components,
               perplexity=perplexity,
               early_exaggeration=early_exaggeration,
               learning_rate=learning_rate,
               n_iter=n_iter,
               n_iter_without_progress=n_iter_without_progress,
               min_grad_norm=min_grad_norm,
               metric=metric,
               init=init,
               verbose=verbose,
               random_state=random_state,
               method=method,
               angle=angle,
               n_jobs=n_jobs)

    x = reg.fit_transform(x).tolist()
    if label_index is not None:
        x = np.insert(x, len(x[0]), values=y, axis=1).tolist()
    return {'array': x,
            'embedding_': reg.embedding_.tolist(),
            'kl_divergence_': reg.kl_divergence_,
            'n_iter_': reg.n_iter_}
