from sklearn.linear_model import ElasticNet


def run(x_train, y_train, x_test, y_test,
        alpha, l1_ratio, fit_intercept, normalize, precompute, max_iter, copy_X, tol, warm_start, positive,
        random_state, selection
        ):
    reg = ElasticNet(alpha=alpha,
                     l1_ratio=l1_ratio,
                     fit_intercept=fit_intercept,
                     normalize=normalize,
                     precompute=precompute,
                     max_iter=max_iter,
                     copy_X=copy_X,
                     tol=tol,
                     warm_start=warm_start,
                     positive=positive,
                     random_state=random_state,
                     selection=selection).fit(x_train, y_train)
    return {'train_predict': reg.predict(x_train).tolist(),
            'test_predict': reg.predict(x_test).tolist(),
            'train_score': reg.score(x_train, y_train),
            'test_score': reg.score(x_test, y_test),
            'coef_': reg.coef_.tolist(),
            'sparse_coef_': reg.sparse_coef_.todense().tolist(),
            'intercept_': reg.intercept_,
            'n_iter_': reg.n_iter_}
