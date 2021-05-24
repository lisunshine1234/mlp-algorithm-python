from sklearn.kernel_ridge import KernelRidge


def run(x_train, y_train, x_test, y_test, alpha, kernel, gamma, degree, coef0, kernel_params
        ):
    reg = KernelRidge(alpha=alpha,
                      kernel=kernel,
                      gamma=gamma,
                      degree=degree,
                      coef0=coef0,
                      kernel_params=kernel_params).fit(x_train, y_train)
    return {'train_predict': reg.predict(x_train).tolist(),
            'test_predict': reg.predict(x_test).tolist(),
            'train_score': reg.score(x_train, y_train),
            'test_score': reg.score(x_test, y_test),
            'dual_coef_': reg.dual_coef_.tolist(),
            'X_fit_': reg.X_fit_.tolist()}
