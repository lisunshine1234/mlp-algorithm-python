from sklearn.gaussian_process import GaussianProcessRegressor


def run(x_train, y_train, x_test, y_test,
        kernel, alpha, optimizer, n_restarts_optimizer, normalize_y, copy_X_train, random_state
        ):
    reg = GaussianProcessRegressor(kernel=kernel,
                                   alpha=alpha,
                                   optimizer=optimizer,
                                   n_restarts_optimizer=n_restarts_optimizer,
                                   normalize_y=normalize_y,
                                   copy_X_train=copy_X_train,
                                   random_state=random_state).fit(x_train, y_train)

    return {'train_predict': reg.predict(x_train).tolist(),
            'test_predict': reg.predict(x_test).tolist(),
            'train_score': reg.score(x_train, y_train),
            'test_score': reg.score(x_test, y_test),
            'X_train_': reg.X_train_.tolist(),
            'y_train_': reg.y_train_.tolist(),
            'L_': reg.L_.tolist(),
            'alpha_': reg.alpha_.tolist(),
            'log_marginal_likelihood_value_': reg.log_marginal_likelihood_value_}
