from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor


def run(x_train, y_train, x_test, y_test,
        base_estimator, estimator_params, n_estimators, learning_rate, loss, random_state):
    base_estimator = getEstimator(base_estimator, estimator_params)
    reg = AdaBoostRegressor(base_estimator=base_estimator,
                            n_estimators=n_estimators,
                            learning_rate=learning_rate,
                            loss=loss,
                            random_state=random_state).fit(x_train, y_train)

    return {'train_predict': reg.predict(x_train).tolist(),
            'test_predict': reg.predict(x_test).tolist(),
            'train_score': reg.score(x_train, y_train),
            'test_score': reg.score(x_test, y_test),
            'estimator_weights_': reg.estimator_weights_.tolist(),
            'estimator_errors_': reg.estimator_errors_.tolist(),
            'feature_importances_': reg.feature_importances_.tolist()
            }


def getEstimator(base_estimator, estimator_params):
    if base_estimator is None:
        return base_estimator
    base_estimator.replace("(", "").replace(")", "")
    if estimator_params is None:
        estimator_params = {}
    return {
        'GradientBoostingRegressor': GradientBoostingRegressor(*estimator_params),
        'ExtraTreesRegressor': ExtraTreesRegressor(*estimator_params),
        'RandomForestRegressor': RandomForestRegressor(*estimator_params)
    }.get(base_estimator, RandomForestRegressor(max_depth=3))
