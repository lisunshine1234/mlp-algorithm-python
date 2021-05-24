from sklearn.cluster import SpectralBiclustering


def run(x, y,
        n_clusters, method, n_components, n_best, svd_method, n_svd_vecs, mini_batch, init, n_init, random_state):
    reg = SpectralBiclustering(n_clusters=n_clusters,
                               method=method,
                               n_components=n_components,
                               n_best=n_best,
                               svd_method=svd_method,
                               n_svd_vecs=n_svd_vecs,
                               mini_batch=mini_batch,
                               init=init,
                               n_init=n_init,
                               random_state=random_state).fit(x, y)
    return {'rows_': reg.rows_.tolist(),
            'columns_': reg.columns_.tolist(),
            'row_labels_': reg.row_labels_.tolist(),
            'column_labels_': reg.column_labels_.tolist()
            }
