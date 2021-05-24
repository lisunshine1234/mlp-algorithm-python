from sklearn.linear_model import LogisticRegression


def run(x_train, y_train, x_test, y_test,
        penalty, dual, tol, C, fit_intercept, intercept_scaling, class_weight, random_state, solver, max_iter, multi_class, verbose, warm_start, n_jobs,
        l1_ratio
        ):
    reg = LogisticRegression(penalty=penalty,
                             dual=dual,
                             tol=tol,
                             C=C,
                             fit_intercept=fit_intercept,
                             intercept_scaling=intercept_scaling,
                             class_weight=class_weight,
                             random_state=random_state,
                             solver=solver,
                             max_iter=max_iter,
                             multi_class=multi_class,
                             verbose=verbose,
                             warm_start=warm_start,
                             n_jobs=n_jobs,
                             l1_ratio=l1_ratio).fit(x_train, y_train)
    return {'train_predict': reg.predict(x_train).tolist(),
            'test_predict': reg.predict(x_test).tolist(),
            'train_score': reg.score(x_train, y_train),
            'test_score': reg.score(x_test, y_test),
            'classes_': reg.classes_.tolist(),
            'coef_': reg.coef_.tolist(),
            'intercept_': reg.intercept_.tolist(),
            'n_iter_': reg.n_iter_.tolist()}
