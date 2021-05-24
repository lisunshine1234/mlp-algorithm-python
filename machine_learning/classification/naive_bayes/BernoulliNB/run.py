from sklearn.naive_bayes import BernoulliNB


def run(x_train, y_train, x_test, y_test,
        alpha, binarize, fit_prior, class_prior):
    reg = BernoulliNB(
        alpha=alpha,
        binarize=binarize,
        fit_prior=fit_prior,
        class_prior=class_prior).fit(x_train, y_train)

    return {'train_predict': reg.predict(x_train).tolist(),
            'test_predict': reg.predict(x_test).tolist(),
            'train_score': reg.score(x_train, y_train),
            'test_score': reg.score(x_test, y_test),
            'class_count_': reg.class_count_.tolist(),
            'class_log_prior_': reg.class_log_prior_.tolist(),
            'classes_': reg.classes_.tolist(),
            'feature_count_': reg.feature_count_.tolist(),
            'feature_log_prob_': reg.feature_log_prob_.tolist(),
            'n_features_': reg.n_features_}
