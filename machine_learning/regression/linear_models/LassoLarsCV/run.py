from sklearn.linear_model import LassoLarsCV


def run(x_train, y_train, x_test, y_test, fit_intercept, verbose, max_iter, normalize, precompute, cv, max_n_alphas, n_jobs, eps, copy_X, positive
        ):
    reg = LassoLarsCV(fit_intercept=fit_intercept,
                      verbose=verbose,
                      max_iter=max_iter,
                      normalize=normalize,
                      precompute=precompute,
                      cv=cv,
                      max_n_alphas=max_n_alphas,
                      n_jobs=n_jobs,
                      eps=eps,
                      copy_X=copy_X,
                      positive=positive
                      ).fit(x_train, y_train)
    return {'train_predict': reg.predict(x_train).tolist(),
            'test_predict': reg.predict(x_test).tolist(),
            'train_score': reg.score(x_train, y_train),
            'test_score': reg.score(x_test, y_test),
            'coef_': reg.coef_.tolist(),
            'intercept_': reg.intercept_,
            'coef_path_': reg.coef_path_.tolist(),
            'alpha_': reg.alpha_,
            'alphas_': reg.alphas_.tolist(),
            'cv_alphas_': reg.cv_alphas_.tolist(),
            'mse_path_': reg.mse_path_.tolist(),
            'n_iter_': reg.n_iter_}
