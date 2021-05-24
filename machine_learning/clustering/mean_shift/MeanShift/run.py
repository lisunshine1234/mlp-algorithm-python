from sklearn.cluster import MeanShift


def run(x_train, y_train, x_test,
        bandwidth, seeds, bin_seeding, min_bin_freq, cluster_all, n_jobs, max_iter
        ):
    reg = MeanShift(bandwidth=bandwidth,
                    seeds=seeds,
                    bin_seeding=bin_seeding,
                    min_bin_freq=min_bin_freq,
                    cluster_all=cluster_all,
                    n_jobs=n_jobs,
                    max_iter=max_iter).fit(x_train, y_train)
    return {'train_predict': reg.predict(x_train).tolist(),
            'test_predict': reg.predict(x_test).tolist(),
            'cluster_centers_': reg.cluster_centers_.tolist(),
            'labels_': reg.labels_.tolist(),
            'n_iter_': reg.n_iter_}
