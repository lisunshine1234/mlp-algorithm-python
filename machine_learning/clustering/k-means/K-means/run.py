from sklearn.cluster import KMeans


def run(x_train, y_train, x_test, y_test,
        n_clusters, init, n_init, max_iter, tol, verbose, random_state, copy_x, algorithm
        ):
    reg = KMeans(n_clusters=n_clusters,
                 init=init,
                 n_init=n_init,
                 max_iter=max_iter,
                 tol=tol,
                 verbose=verbose,
                 random_state=random_state,
                 copy_x=copy_x,
                 algorithm=algorithm).fit(x_train, y_train)
    return {'train_predict': reg.predict(x_train).tolist(),
            'test_predict': reg.predict(x_test).tolist(),
            'train_score': reg.score(x_train, y_train),
            'test_score': reg.score(x_test, y_test),
            'cluster_centers_': reg.cluster_centers_.tolist(),
            'labels_': reg.labels_.tolist(),
            'inertia_': reg.inertia_,
            'n_iter_': reg.n_iter_}
