from sklearn.manifold import SpectralEmbedding
import numpy as np

def run(array,label_index, n_components, affinity, gamma, random_state, eigen_solver, n_neighbors, n_jobs
        ):
    x = np.array(array)
    if label_index is not None:
        if label_index < 0:
            label_index = len(array[0]) + label_index

        y = x[:, label_index]
        x = np.delete(array, label_index, axis=1)
    reg = SpectralEmbedding(n_components=n_components,
                            affinity=affinity,
                            gamma=gamma,
                            random_state=random_state,
                            eigen_solver=eigen_solver,
                            n_neighbors=n_neighbors,
                            n_jobs=n_jobs)

    x = reg.fit_transform(x).tolist()
    if label_index is not None:
        x = np.insert(x, len(x[0]), values=y, axis=1).tolist()
    return {'array': x,
            'embedding_': reg.embedding_.tolist(),
            'affinity_matrix_': reg.affinity_matrix_.todense().tolist(),
            'n_neighbors_': reg.n_neighbors_}
