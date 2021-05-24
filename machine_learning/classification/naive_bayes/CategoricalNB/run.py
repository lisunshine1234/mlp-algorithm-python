from sklearn.naive_bayes import CategoricalNB


def run(x_train, y_train, x_test, y_test, alpha, fit_prior, class_prior):
    reg = CategoricalNB(alpha=alpha,
                        fit_prior=fit_prior,
                        class_prior=class_prior).fit(x_train, y_train)

    category_count_ = []
    for i in reg.category_count_:
        category_count_.append(i.tolist())
    feature_log_prob_ = []
    for i in reg.feature_log_prob_:
        feature_log_prob_.append(i.tolist())
    return {'train_predict': reg.predict(x_train).tolist(),
            'test_predict': reg.predict(x_test).tolist(),
            'train_score': reg.score(x_train, y_train),
            'test_score': reg.score(x_test, y_test),
            'category_count_': category_count_,
            'class_count_': reg.class_count_.tolist(),
            'class_log_prior_': reg.class_log_prior_.tolist(),
            'classes_': reg.classes_.tolist(),
            'feature_log_prob_': feature_log_prob_,
            'n_features_': reg.n_features_}
