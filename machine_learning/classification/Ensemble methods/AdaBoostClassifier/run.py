from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import ARDRegression, BayesianRidge, ElasticNet, ElasticNetCV, HuberRegressor, Lars, LarsCV, \
    Lasso, LassoCV, LassoLars, LassoLarsCV, LassoLarsIC, LinearRegression, OrthogonalMatchingPursuit, \
    OrthogonalMatchingPursuitCV, PassiveAggressiveClassifier, PassiveAggressiveRegressor, Perceptron, \
    Ridge, RidgeClassifier, RidgeClassifierCV, RidgeCV, SGDClassifier, SGDRegressor, TheilSenRegressor
from sklearn.svm import SVR, SVC, NuSVR, OneClassSVM
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier

def run(x_train, y_train, x_test, y_test,
        base_estimator, n_estimators, learning_rate, algorithm, random_state,estimator_params
        ):
    base_estimator = get_estimator(base_estimator, estimator_params)

    reg = AdaBoostClassifier(base_estimator=base_estimator,
                             n_estimators=n_estimators,
                             learning_rate=learning_rate,
                             algorithm=algorithm,
                             random_state=random_state).fit(x_train, y_train)

    return {'train_predict': reg.predict(x_train).tolist(),
            'test_predict': reg.predict(x_test).tolist(),
            'train_score': reg.score(x_train, y_train),
            'test_score': reg.score(x_test, y_test),
            'classes_': reg.classes_.tolist(),
            'n_classes_': reg.n_classes_,
            'estimator_weights_': reg.estimator_weights_.tolist(),
            'estimator_errors_': reg.estimator_errors_.tolist(),
            'feature_importances_': reg.feature_importances_.tolist()}



def get_estimator(base_estimator, estimator_params):
    if base_estimator is None:
        return base_estimator
    base_estimator.replace("(", "").replace(")", "")
    if estimator_params is None:
        estimator_params = {}
    return {
        'GradientBoostingClassifier': GradientBoostingClassifier(*estimator_params),
        'ExtraTreesClassifier': ExtraTreesClassifier(*estimator_params),
        'RandomForestClassifier': RandomForestClassifier(*estimator_params)
    }.get(base_estimator, None)