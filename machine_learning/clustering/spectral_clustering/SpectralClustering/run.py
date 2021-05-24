from sklearn.cluster import SpectralClustering


def run(x_train, y_train,
        n_clusters, eigen_solver, n_components, random_state, n_init, gamma, affinity, n_neighbors, eigen_tol, assign_labels, degree, coef0, kernel_params,
        n_jobs
        ):
    reg = SpectralClustering(n_clusters=n_clusters,
                             eigen_solver=eigen_solver,
                             n_components=n_components,
                             random_state=random_state,
                             n_init=n_init,
                             gamma=gamma,
                             affinity=affinity,
                             n_neighbors=n_neighbors,
                             eigen_tol=eigen_tol,
                             assign_labels=assign_labels,
                             degree=degree,
                             coef0=coef0,
                             kernel_params=kernel_params,
                             n_jobs=n_jobs).fit(x_train, y_train)
    return {'affinity_matrix_': reg.affinity_matrix_.tolist(),
            'labels_': reg.labels_.tolist()}
