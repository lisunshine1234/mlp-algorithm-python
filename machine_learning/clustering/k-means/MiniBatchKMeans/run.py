from sklearn.cluster import MiniBatchKMeans


def run(x_train, y_train, x_test, y_test,
        n_clusters, init, max_iter, batch_size, verbose, compute_labels, random_state, tol, max_no_improvement, init_size, n_init, reassignment_ratio
        ):
    reg = MiniBatchKMeans(n_clusters=n_clusters,
                          init=init,
                          max_iter=max_iter,
                          batch_size=batch_size,
                          verbose=verbose,
                          compute_labels=compute_labels,
                          random_state=random_state,
                          tol=tol,
                          max_no_improvement=max_no_improvement,
                          init_size=init_size,
                          n_init=n_init,
                          reassignment_ratio=reassignment_ratio
                          ).fit(x_train, y_train)
    labels_ = None
    if compute_labels:
        labels_ = reg.labels_
    return {'train_predict': reg.predict(x_train).tolist(),
            'test_predict': reg.predict(x_test).tolist(),
            'train_score': reg.score(x_train, y_train),
            'test_score': reg.score(x_test, y_test),
            'cluster_centers_': reg.cluster_centers_.tolist(),
            'labels_': labels_.tolist(),
            'inertia_': reg.inertia_
            }
