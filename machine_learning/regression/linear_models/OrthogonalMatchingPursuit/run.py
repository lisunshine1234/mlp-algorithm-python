from sklearn.linear_model import OrthogonalMatchingPursuit


def run(x_train, y_train, x_test, y_test,
        n_nonzero_coefs, tol, fit_intercept, normalize, precompute
        ):
    reg = OrthogonalMatchingPursuit(
        n_nonzero_coefs=n_nonzero_coefs,
        tol=tol,
        fit_intercept=fit_intercept,
        normalize=normalize,
        precompute=precompute).fit(x_train, y_train)
    return {'train_predict': reg.predict(x_train).tolist(),
            'test_predict': reg.predict(x_test).tolist(),
            'train_score': reg.score(x_train, y_train),
            'test_score': reg.score(x_test, y_test),
            'coef_': reg.coef_.tolist(),
            'intercept_': reg.intercept_,
            'n_iter_': reg.n_iter_
            }
