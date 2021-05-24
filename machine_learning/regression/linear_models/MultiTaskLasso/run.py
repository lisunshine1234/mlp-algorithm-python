from sklearn.linear_model import MultiTaskLasso


def run(x_train, y_train, x_test, y_test,
        alpha, fit_intercept, normalize, copy_X, max_iter, tol, warm_start, random_state, selection
        ):
    reg = MultiTaskLasso(alpha=alpha,
                         fit_intercept=fit_intercept,
                         normalize=normalize,
                         copy_X=copy_X,
                         max_iter=max_iter,
                         tol=tol,
                         warm_start=warm_start,
                         random_state=random_state,
                         selection=selection).fit(x_train, y_train)
    return {'train_predict': reg.predict(x_train).tolist(),
            'test_predict': reg.predict(x_test).tolist(),
            'train_score': reg.score(x_train, y_train),
            'test_score': reg.score(x_test, y_test),
            'coef_': reg.coef_.tolist(),
            'intercept_': reg.intercept_.tolist(),
            'n_iter_': reg.n_iter_
            }
