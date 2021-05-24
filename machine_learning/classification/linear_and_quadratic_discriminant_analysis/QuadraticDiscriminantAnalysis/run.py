from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis


def run(x_train, y_train, x_test, y_test, priors, reg_param, store_covariance, tol):
    reg = QuadraticDiscriminantAnalysis(priors=priors,
                                        reg_param=reg_param,
                                        store_covariance=store_covariance,
                                        tol=tol
                                        ).fit(x_train, y_train)
    covariance_ = None
    if store_covariance:
        covariance_ = []
        for i in reg.rotations_:
            covariance_.append(i.tolist())
    rotations_ = []
    for i in reg.rotations_:
        rotations_.append(i.tolist())
    scalings_ = []
    for i in reg.scalings_:
        scalings_.append(i.tolist())

    return {'train_predict': reg.predict(x_train).tolist(),
            'test_predict': reg.predict(x_test).tolist(),
            'train_score': reg.score(x_train, y_train),
            'test_score': reg.score(x_test, y_test),
            'means_': reg.means_.tolist(),
            'priors_': reg.priors_.tolist(),
            'covariance_': covariance_,
            'rotations_': rotations_,
            'scalings_': scalings_,
            'classes_': reg.classes_.tolist()}
