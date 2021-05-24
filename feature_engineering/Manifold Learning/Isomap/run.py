from sklearn.manifold import Isomap
import numpy as np

def run(array, label_index,n_neighbors, n_components, eigen_solver, tol, max_iter, path_method, neighbors_algorithm, n_jobs, metric, p, metric_params
        ):
    x = np.array(array)
    if label_index is not None:
        if label_index < 0:
            label_index = len(array[0]) + label_index

        y = x[:, label_index]
        x = np.delete(array, label_index, axis=1)
    reg = Isomap(n_neighbors=n_neighbors,
                 n_components=n_components,
                 eigen_solver=eigen_solver,
                 tol=tol,
                 max_iter=max_iter,
                 path_method=path_method,
                 neighbors_algorithm=neighbors_algorithm,
                 n_jobs=n_jobs,
                 metric=metric,
                 p=p,
                 metric_params=metric_params)
    x = reg.fit_transform(x).tolist()
    if label_index is not None:
        x = np.insert(x, len(x[0]), values=y, axis=1).tolist()
    return {'array': x,
            'embedding_': reg.embedding_.tolist(),
            'dist_matrix_': reg.dist_matrix_.tolist()}
