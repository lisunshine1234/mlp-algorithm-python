from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis


def run(x_train, y_train, x_test, y_test,
        solver, shrinkage, priors, n_components, store_covariance, tol
        ):
    reg = LinearDiscriminantAnalysis(solver=solver,
                                     shrinkage=shrinkage,
                                     priors=priors,
                                     n_components=n_components,
                                     store_covariance=store_covariance,
                                     tol=tol).fit(x_train, y_train)
    covariance_ = None
    if store_covariance:
        covariance_ = reg.covariance_.tolist()
    return {'train_predict': reg.predict(x_train).tolist(),
            'test_predict': reg.predict(x_test).tolist(),
            'train_score': reg.score(x_train, y_train),
            'test_score': reg.score(x_test, y_test),
            'coef_': reg.coef_.tolist(),
            'intercept_': reg.intercept_.tolist(),
            'covariance_': covariance_,
            'explained_variance_ratio_': reg.explained_variance_ratio_.tolist(),
            'means_': reg.means_.tolist(),
            'priors_': reg.priors_.tolist(),
            'scalings_': reg.scalings_.tolist(),
            'xbar_': reg.xbar_.tolist(),
            'classes_': reg.classes_.tolist()}
