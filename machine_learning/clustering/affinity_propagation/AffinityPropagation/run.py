from sklearn.cluster import AffinityPropagation


def run(x_train, y_train, x_test,
        damping, max_iter, convergence_iter, copy, preference, affinity, verbose, random_state ):
    reg = AffinityPropagation(damping=damping,
                              max_iter=max_iter,
                              convergence_iter=convergence_iter,
                              copy=copy,
                              preference=preference,
                              affinity=affinity,
                              verbose=verbose,
                              random_state=random_state).fit(x_train, y_train)
    return {'train_predict': reg.predict(x_train).tolist(),
            'test_predict': reg.predict(x_test).tolist(),
            'cluster_centers_indices_': reg.cluster_centers_indices_.tolist(),
            'cluster_centers_': reg.cluster_centers_.tolist(),
            'labels_': reg.labels_.tolist(),
            'affinity_matrix_': reg.affinity_matrix_.tolist(),
            'n_iter_': reg.n_iter_
            }
