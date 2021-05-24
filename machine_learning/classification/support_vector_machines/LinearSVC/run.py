from sklearn.svm import LinearSVC, LinearSVR, NuSVC, NuSVR, OneClassSVM, SVC, SVR


def run(x_train, y_train, x_test, y_test,
        penalty, loss, dual, tol, C, multi_class, fit_intercept, intercept_scaling, class_weight, verbose, random_state, max_iter
        ):
    reg = LinearSVC(penalty=penalty,
                    loss=loss,
                    dual=dual,
                    tol=tol,
                    C=C,
                    multi_class=multi_class,
                    fit_intercept=fit_intercept,
                    intercept_scaling=intercept_scaling,
                    class_weight=class_weight,
                    verbose=verbose,
                    random_state=random_state,
                    max_iter=max_iter).fit(x_train, y_train)
    return {'train_predict': reg.predict(x_train).tolist(),
            'test_predict': reg.predict(x_test).tolist(),
            'train_score': reg.score(x_train, y_train),
            'test_score': reg.score(x_test, y_test),
            'coef_': reg.coef_.tolist(),
            'intercept_': reg.intercept_.tolist(),
            'classes_': reg.classes_.tolist(),
            'n_iter_': int(reg.n_iter_)}
