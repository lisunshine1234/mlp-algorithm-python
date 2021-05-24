from sklearn.neighbors import RadiusNeighborsRegressor


def run(x_train, y_train, x_test, y_test,
        radius, weights, algorithm, leaf_size, p, metric, metric_params, n_jobs,
        **kwargs):
    reg = RadiusNeighborsRegressor(radius=radius,
                                   weights=weights,
                                   algorithm=algorithm,
                                   leaf_size=leaf_size,
                                   p=p,
                                   metric=metric,
                                   metric_params=metric_params,
                                   n_jobs=n_jobs,
                                   **kwargs).fit(x_train, y_train)

    return {'train_predict': reg.predict(x_train).tolist(),
            'test_predict': reg.predict(x_test).tolist(),
            'train_score': reg.score(x_train, y_train),
            'test_score': reg.score(x_test, y_test),
            'effective_metric_': reg.effective_metric_,
            'effective_metric_params_': reg.effective_metric_params_}
