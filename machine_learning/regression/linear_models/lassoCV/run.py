from sklearn.linear_model import LassoCV


def run(x_train, y_train, x_test, y_test, eps, n_alphas, alphas, fit_intercept, normalize, precompute, max_iter, tol, copy_X, cv, verbose, n_jobs,
        positive, random_state, selection):
    reg = LassoCV(eps=eps,
                  n_alphas=n_alphas,
                  alphas=alphas,
                  fit_intercept=fit_intercept,
                  normalize=normalize,
                  precompute=precompute,
                  max_iter=max_iter,
                  tol=tol,
                  copy_X=copy_X,
                  cv=cv,
                  verbose=verbose,
                  n_jobs=n_jobs,
                  positive=positive,
                  random_state=random_state,
                  selection=selection).fit(x_train, y_train)
    return {'train_predict': reg.predict(x_train).tolist(),
            'test_predict': reg.predict(x_test).tolist(),
            'train_score': reg.score(x_train, y_train),
            'test_score': reg.score(x_test, y_test),
            'alpha_': reg.alpha_,
            'coef_': reg.coef_.tolist(),
            'intercept_': reg.intercept_,
            'mse_path_': reg.mse_path_.tolist(),
            'alphas_': reg.alphas_.tolist(),
            'dual_gap_': reg.dual_gap_.tolist(),
            'n_iter_': reg.n_iter_}
