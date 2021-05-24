from sklearn.linear_model import TweedieRegressor


def run(x_train, y_train, x_test, y_test,
        power, alpha, fit_intercept, link, max_iter, tol, warm_start, verbose):
    reg = TweedieRegressor(power=power,
                           alpha=alpha,
                           fit_intercept=fit_intercept,
                           link=link,
                           max_iter=max_iter,
                           tol=tol,
                           warm_start=warm_start,
                           verbose=verbose).fit(x_train, y_train)
    return {'train_predict': reg.predict(x_train).tolist(),
            'test_predict': reg.predict(x_test).tolist(),
            'train_score': reg.score(x_train, y_train),
            'test_score': reg.score(x_test, y_test),
            'coef_': reg.coef_.tolist(),
            'intercept_': reg.intercept_,
            'n_iter_': reg.n_iter_}
