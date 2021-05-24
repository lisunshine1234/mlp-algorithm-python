from sklearn.svm import LinearSVC, LinearSVR, NuSVC, NuSVR, OneClassSVM, SVC, SVR


def run(x_train, y_train, x_test, y_test,
        epsilon, tol, C, loss, fit_intercept, intercept_scaling, dual, verbose, random_state, max_iter
        ):
    reg = LinearSVR(epsilon=epsilon,
                    tol=tol,
                    C=C,
                    loss=loss,
                    fit_intercept=fit_intercept,
                    intercept_scaling=intercept_scaling,
                    dual=dual,
                    verbose=verbose,
                    random_state=random_state,
                    max_iter=max_iter).fit(x_train, y_train)
    return {'train_predict': reg.predict(x_train).tolist(),
            'test_predict': reg.predict(x_test).tolist(),
            'train_score': reg.score(x_train, y_train),
            'test_score': reg.score(x_test, y_test),
            'coef_': reg.coef_.tolist(),
            'intercept_': reg.intercept_.tolist(),
            'n_iter_': int(reg.n_iter_)}
