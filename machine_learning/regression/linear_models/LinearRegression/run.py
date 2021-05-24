from sklearn.linear_model import LinearRegression


def run(x_train, y_train, x_test, y_test, fit_intercept,
        normalize,
        copy_X,
        n_jobs):
    reg = LinearRegression(fit_intercept=fit_intercept,
                           normalize=normalize,
                           copy_X=copy_X,
                           n_jobs=n_jobs).fit(x_train, y_train)
    return {'train_predict': reg.predict(x_train).tolist(),
            'test_predict': reg.predict(x_test).tolist(),
            'train_score': reg.score(x_train, y_train),
            'test_score': reg.score(x_test, y_test),
            'coef_': reg.coef_.tolist(),
            'rank_': reg.rank_,
            'singular_': reg.singular_.tolist(),
            'intercept_': reg.intercept_}
