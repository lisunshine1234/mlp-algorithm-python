from sklearn.linear_model import Ridge


def run(x_train, y_train, x_test, y_test, alpha,
        fit_intercept,
        normalize,
        copy_X,
        max_iter,
        tol,
        solver,
        random_state):
    reg = Ridge(alpha=alpha,
                fit_intercept=fit_intercept,
                normalize=normalize,
                copy_X=copy_X,
                max_iter=max_iter,
                tol=tol,
                solver=solver,
                random_state=random_state).fit(x_train, y_train)
    n_iter_ = None
    if reg.n_iter_ is not None:
        n_iter_ = reg.n_iter_.tolist()
    return {'train_predict': reg.predict(x_train).tolist(),
            'test_predict': reg.predict(x_test).tolist(),
            'train_score': reg.score(x_train, y_train),
            'test_score': reg.score(x_test, y_test),
            'coef_': reg.coef_.tolist(),
            'intercept_': reg.intercept_,
            'n_iter_': n_iter_}
