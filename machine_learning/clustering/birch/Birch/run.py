from sklearn.cluster import Birch


def run(x_train, y_train, x_test,
        threshold, branching_factor, n_clusters, compute_labels, copy
        ):
    reg = Birch(threshold=threshold,
                branching_factor=branching_factor,
                n_clusters=n_clusters,
                compute_labels=compute_labels,
                copy=copy).fit(x_train, y_train)
    return {'train_predict': reg.predict(x_train).tolist(),
            'test_predict': reg.predict(x_test).tolist(),
            'subcluster_centers_': reg.subcluster_centers_.tolist(),
            'subcluster_labels_': reg.subcluster_labels_.tolist(),
            'labels_': reg.labels_.tolist()}
