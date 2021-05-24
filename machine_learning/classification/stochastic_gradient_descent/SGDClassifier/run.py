from sklearn.linear_model import SGDClassifier


def run(x_train, y_train, x_test, y_test,
        loss, penalty, alpha, l1_ratio, fit_intercept, max_iter, tol, shuffle, verbose, epsilon, n_jobs, random_state,
        learning_rate, eta0, power_t, early_stopping, validation_fraction, n_iter_no_change, class_weight, warm_start,
        average
        ):
    reg = SGDClassifier(

        loss=loss,
        penalty=penalty,
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=fit_intercept,
        max_iter=max_iter,
        tol=tol,
        shuffle=shuffle,
        verbose=verbose,
        epsilon=epsilon,
        n_jobs=n_jobs,
        random_state=random_state,
        learning_rate=learning_rate,
        eta0=eta0,
        power_t=power_t,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        n_iter_no_change=n_iter_no_change,
        class_weight=class_weight,
        warm_start=warm_start,
        average=average).fit(x_train, y_train)
    return {'train_predict': reg.predict(x_train).tolist(),
            'test_predict': reg.predict(x_test).tolist(),
            'train_score': reg.score(x_train, y_train),
            'test_score': reg.score(x_test, y_test),
            'coef_': reg.coef_.tolist(),
            'intercept_': reg.intercept_.tolist(),
            'n_iter_': reg.n_iter_,
            'classes_': reg.classes_.tolist(),
            't_': reg.t_}
