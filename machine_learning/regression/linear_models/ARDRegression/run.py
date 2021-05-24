from sklearn.linear_model import ARDRegression


def run(x_train, y_train, x_test, y_test, n_iter, tol, alpha_1, alpha_2, lambda_1, lambda_2, compute_score, threshold_lambda, fit_intercept,
        normalize, copy_X, verbose
        ):
    reg = ARDRegression(n_iter=n_iter,
                        tol=tol,
                        alpha_1=alpha_1,
                        alpha_2=alpha_2,
                        lambda_1=lambda_1,
                        lambda_2=lambda_2,
                        compute_score=compute_score,
                        threshold_lambda=threshold_lambda,
                        fit_intercept=fit_intercept,
                        normalize=normalize,
                        copy_X=copy_X,
                        verbose=verbose).fit(x_train, y_train)
    return {'train_predict': reg.predict(x_train).tolist(),
            'test_predict': reg.predict(x_test).tolist(),
            'train_score': reg.score(x_train, y_train),
            'test_score': reg.score(x_test, y_test),
            'coef_': reg.coef_.tolist(),
            'alpha_': reg.alpha_,
            'lambda_': reg.lambda_.tolist(),
            'sigma_': reg.sigma_.tolist(),
            'scores_': reg.scores_,
            'intercept_': reg.intercept_}
