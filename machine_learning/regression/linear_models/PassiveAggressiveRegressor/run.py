from sklearn.linear_model import PassiveAggressiveRegressor


def run(x_train, y_train, x_test, y_test,
        C, fit_intercept, max_iter, tol, early_stopping, validation_fraction, n_iter_no_change, shuffle, verbose, loss, epsilon, random_state, warm_start,
        average
        ):
    reg = PassiveAggressiveRegressor(C=C,
                                     fit_intercept=fit_intercept,
                                     max_iter=max_iter,
                                     tol=tol,
                                     early_stopping=early_stopping,
                                     validation_fraction=validation_fraction,
                                     n_iter_no_change=n_iter_no_change,
                                     shuffle=shuffle,
                                     verbose=verbose,
                                     loss=loss,
                                     epsilon=epsilon,
                                     random_state=random_state,
                                     warm_start=warm_start,
                                     average=average).fit(x_train, y_train)
    return {'train_predict': reg.predict(x_train).tolist(),
            'test_predict': reg.predict(x_test).tolist(),
            'train_score': reg.score(x_train, y_train),
            'test_score': reg.score(x_test, y_test),
            'coef_': reg.coef_.tolist(),
            'intercept_': reg.intercept_.tolist(),
            'n_iter_': reg.n_iter_,
            't_': reg.t_}
