from sklearn.gaussian_process import GaussianProcessClassifier


def run(x_train, y_train, x_test, y_test,
        kernel, optimizer, n_restarts_optimizer, max_iter_predict, warm_start, copy_X_train, random_state, multi_class, n_jobs):
    reg = GaussianProcessClassifier(kernel=kernel,
                                    optimizer=optimizer,
                                    n_restarts_optimizer=n_restarts_optimizer,
                                    max_iter_predict=max_iter_predict,
                                    warm_start=warm_start,
                                    copy_X_train=copy_X_train,
                                    random_state=random_state,
                                    multi_class=multi_class,
                                    n_jobs=n_jobs).fit(x_train, y_train)

    return {'train_predict': reg.predict(x_train).tolist(),
            'test_predict': reg.predict(x_test).tolist(),
            'train_score': reg.score(x_train, y_train),
            'test_score': reg.score(x_test, y_test),
            'log_marginal_likelihood_value_': reg.log_marginal_likelihood_value_,
            'classes_': reg.classes_.tolist(),
            'n_classes_': reg.n_classes_}
