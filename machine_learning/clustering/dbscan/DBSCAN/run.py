from sklearn.cluster import DBSCAN


def run(x_train, y_train,
        eps, min_samples, metric, metric_params, algorithm, leaf_size, p, n_jobs
        ):
    reg = DBSCAN(eps=eps,
                 min_samples=min_samples,
                 metric=metric,
                 metric_params=metric_params,
                 algorithm=algorithm,
                 leaf_size=leaf_size,
                 p=p,
                 n_jobs=n_jobs).fit(x_train, y_train)
    return {
        'core_sample_indices_': reg.core_sample_indices_.tolist(),
        'components_': reg.components_.tolist(),
        'labels_': reg.labels_.tolist()
    }
