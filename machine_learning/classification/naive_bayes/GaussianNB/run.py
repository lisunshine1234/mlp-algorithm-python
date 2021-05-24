from sklearn.naive_bayes import GaussianNB


def run(x_train, y_train, x_test, y_test, priors, var_smoothing):
    reg = GaussianNB(priors=priors,
                     var_smoothing=var_smoothing).fit(x_train, y_train)

    return {'train_predict': reg.predict(x_train).tolist(),
            'test_predict': reg.predict(x_test).tolist(),
            'train_score': reg.score(x_train, y_train),
            'test_score': reg.score(x_test, y_test),
            'class_count_': reg.class_count_.tolist(),
            'class_prior_': reg.class_prior_.tolist(),
            'classes_': reg.classes_.tolist(),
            'epsilon_': reg.epsilon_,
            'sigma_': reg.sigma_.tolist(),
            'theta_': reg.theta_.tolist()}
