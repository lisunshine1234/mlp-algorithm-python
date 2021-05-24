from sklearn.linear_model import Perceptron


def run(x_train, y_train, x_test, y_test,
        penalty, alpha, fit_intercept, max_iter, tol, shuffle, verbose, eta0, n_jobs, random_state, early_stopping, validation_fraction, n_iter_no_change,
        class_weight, warm_start
        ):
    reg = Perceptron(penalty=penalty,
                     alpha=alpha,
                     fit_intercept=fit_intercept,
                     max_iter=max_iter,
                     tol=tol,
                     shuffle=shuffle,
                     verbose=verbose,
                     eta0=eta0,
                     n_jobs=n_jobs,
                     random_state=random_state,
                     early_stopping=early_stopping,
                     validation_fraction=validation_fraction,
                     n_iter_no_change=n_iter_no_change,
                     class_weight=class_weight,
                     warm_start=warm_start
                     ).fit(x_train, y_train)
    return {'train_predict': reg.predict(x_train).tolist(),
            'test_predict': reg.predict(x_test).tolist(),
            'train_score': reg.score(x_train, y_train),
            'test_score': reg.score(x_test, y_test),
            'coef_': reg.coef_.tolist(),
            'intercept_': reg.intercept_.tolist(),
            'n_iter_': reg.n_iter_,
            'classes_': reg.classes_.tolist(),
            't_': reg.t_
            }
