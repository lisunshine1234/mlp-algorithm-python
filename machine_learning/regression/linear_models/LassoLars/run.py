from sklearn.linear_model import LassoLars


def run(x_train, y_train, x_test, y_test, alpha, fit_intercept, verbose, normalize, precompute, max_iter, eps, copy_X, fit_path, positive, jitter, random_state
        ):
    reg = LassoLars(alpha=alpha,
                    fit_intercept=fit_intercept,
                    verbose=verbose,
                    normalize=normalize,
                    precompute=precompute,
                    max_iter=max_iter,
                    eps=eps,
                    copy_X=copy_X,
                    fit_path=fit_path,
                    positive=positive,
                    jitter=jitter,
                    random_state=random_state).fit(x_train, y_train)
    return {'train_predict': reg.predict(x_train).tolist(),
            'test_predict': reg.predict(x_test).tolist(),
            'train_score': reg.score(x_train, y_train),
            'test_score': reg.score(x_test, y_test),
            'alphas_': reg.alphas_.tolist(),
            'active_': reg.active_,
            'coef_path_': reg.coef_path_.tolist(),
            'coef_': reg.coef_.tolist(),
            'intercept_': reg.intercept_,
            'n_iter_': reg.n_iter_}
