from sklearn.manifold import LocallyLinearEmbedding

import numpy as np

def run(array,label_index, n_neighbors, n_components, reg, eigen_solver, tol, max_iter, method, hessian_tol, modified_tol, neighbors_algorithm, random_state, n_jobs
        ):
    x = np.array(array)
    if label_index is not None:
        if label_index < 0:
            label_index = len(array[0]) + label_index

        y = x[:, label_index]
        x = np.delete(array, label_index, axis=1)
    reg = LocallyLinearEmbedding(n_neighbors=n_neighbors,
                                 n_components=n_components,
                                 reg=reg,
                                 eigen_solver=eigen_solver,
                                 tol=tol,
                                 max_iter=max_iter,
                                 method=method,
                                 hessian_tol=hessian_tol,
                                 modified_tol=modified_tol,
                                 neighbors_algorithm=neighbors_algorithm,
                                 random_state=random_state,
                                 n_jobs=n_jobs)

    x = reg.fit_transform(x).tolist()
    if label_index is not None:
        x = np.insert(x, len(x[0]), values=y, axis=1).tolist()
    return {'array': x,
            'embedding_': reg.embedding_.tolist(),
            'reconstruction_error_': reg.reconstruction_error_
            }
