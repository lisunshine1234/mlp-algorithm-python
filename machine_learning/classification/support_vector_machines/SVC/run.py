from sklearn.svm import LinearSVC, LinearSVR, NuSVC, NuSVR, OneClassSVM, SVC, SVR


def run(x_train, y_train, x_test, y_test,
        C, kernel, degree, gamma, coef0, shrinking, probability, tol, cache_size, class_weight, verbose, max_iter, decision_function_shape, break_ties,
        random_state
        ):
    reg = SVC(C=C,
              kernel=kernel,
              degree=degree,
              gamma=gamma,
              coef0=coef0,
              shrinking=shrinking,
              probability=probability,
              tol=tol,
              cache_size=cache_size,
              class_weight=class_weight,
              verbose=verbose,
              max_iter=max_iter,
              decision_function_shape=decision_function_shape,
              break_ties=break_ties,
              random_state=random_state).fit(x_train, y_train)

    coef_ = None
    if kernel == 'linear':
        coef_ = reg.coef_.tolist()
    return {'train_predict': reg.predict(x_train).tolist(),
            'test_predict': reg.predict(x_test).tolist(),
            'train_score': reg.score(x_train, y_train),
            'test_score': reg.score(x_test, y_test),
            'support_': reg.support_.tolist(),
            'support_vectors_': reg.support_vectors_.tolist(),
            'n_support_': reg.n_support_.tolist(),
            'dual_coef_': reg.dual_coef_.tolist(),
            'coef_': coef_,
            'intercept_': reg.intercept_.tolist(),
            'fit_status_': reg.fit_status_,
            'classes_': reg.classes_.tolist(),
            'probA_': reg.probA_.tolist(),
            'probB_': reg.probB_.tolist(),
            'class_weight_': reg.class_weight_.tolist(),
            'shape_fit_': reg.shape_fit_
            }
