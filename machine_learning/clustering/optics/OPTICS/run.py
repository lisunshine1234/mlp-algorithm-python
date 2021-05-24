from sklearn.cluster import OPTICS


def run(x_train, y_train,
        min_samples, max_eps, metric, p, metric_params, cluster_method, eps, xi, predecessor_correction, min_cluster_size, algorithm, leaf_size, n_jobs
        ):
    reg = OPTICS(min_samples=min_samples,
                 max_eps=max_eps,
                 metric=metric,
                 p=p,
                 metric_params=metric_params,
                 cluster_method=cluster_method,
                 eps=eps,
                 xi=xi,
                 predecessor_correction=predecessor_correction,
                 min_cluster_size=min_cluster_size,
                 algorithm=algorithm,
                 leaf_size=leaf_size,
                 n_jobs=n_jobs
                 ).fit(x_train, y_train)
    return {'labels_': reg.labels_.tolist(),
            'reachability_': reg.reachability_.tolist(),
            'ordering_': reg.ordering_.tolist(),
            'core_distances_': reg.core_distances_.tolist(),
            'predecessor_': reg.predecessor_.tolist(),
            'cluster_hierarchy_': reg.cluster_hierarchy_.tolist()}
