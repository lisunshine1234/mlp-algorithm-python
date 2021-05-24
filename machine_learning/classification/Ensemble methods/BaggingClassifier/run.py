from sklearn.ensemble import BaggingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import ARDRegression, BayesianRidge, ElasticNet, ElasticNetCV, HuberRegressor, Lars, LarsCV, \
    Lasso, LassoCV, LassoLars, LassoLarsCV, LassoLarsIC, LinearRegression, OrthogonalMatchingPursuit, \
    OrthogonalMatchingPursuitCV, PassiveAggressiveClassifier, PassiveAggressiveRegressor, Perceptron, \
    Ridge, RidgeClassifier, RidgeClassifierCV, RidgeCV, SGDClassifier, SGDRegressor, TheilSenRegressor
from sklearn.svm import SVR, SVC, NuSVR, OneClassSVM
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier


def run(x_train, y_train, x_test, y_test,
        base_estimator, estimator_params, n_estimators, max_samples, max_features, bootstrap, bootstrap_features, oob_score, warm_start, n_jobs, random_state,
        verbose
        ):
    base_estimator = get_estimator(base_estimator, estimator_params)
    print(base_estimator)
    reg = BaggingClassifier(base_estimator=base_estimator,
                            n_estimators=n_estimators,
                            max_samples=max_samples,
                            max_features=max_features,
                            bootstrap=bootstrap,
                            bootstrap_features=bootstrap_features,
                            oob_score=oob_score,
                            warm_start=warm_start,
                            n_jobs=n_jobs,
                            random_state=random_state,
                            verbose=verbose
                            ).fit(x_train, y_train)
    oob_score_ = None
    oob_decision_function_ = None
    if oob_score:
        oob_score_ = reg.oob_score_
        oob_decision_function_ = reg.oob_decision_function_.tolist()
    estimators_samples_ = []
    for i in reg.estimators_features_:
        estimators_samples_.append(i.tolist())
    estimators_features_ = []
    for i in reg.estimators_features_:
        estimators_features_.append(i.tolist())
    print(reg.predict(x_train))
    return {'train_predict': reg.predict(x_train).tolist(),
            'test_predict': reg.predict(x_test).tolist(),
            'train_score': reg.score(x_train, y_train),
            'test_score': reg.score(x_test, y_test),
            'n_features_': reg.n_features_,
            'estimators_samples_': estimators_samples_,
            'estimators_features_': estimators_features_,
            'classes_': reg.classes_.tolist(),
            'n_classes_': reg.n_classes_,
            'oob_score_': oob_score_,
            'oob_decision_function_': oob_decision_function_
            }



def get_estimator(base_estimator, estimator_params):
    if base_estimator is None:
        return base_estimator
    base_estimator.replace("(", "").replace(")", "")
    if estimator_params is None:
        estimator_params = {}
    return {

        'SVC': SVC(*estimator_params),
        'OneClassSVM': OneClassSVM(*estimator_params),
        'PassiveAggressiveClassifier': PassiveAggressiveClassifier(*estimator_params),
        'Perceptron': Perceptron(*estimator_params),
        'RidgeClassifier': RidgeClassifier(*estimator_params),
        'RidgeClassifierCV': RidgeClassifierCV(*estimator_params),
        'SGDClassifier': SGDClassifier(*estimator_params),
        'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(*estimator_params),
        'DecisionTreeClassifier': DecisionTreeClassifier(*estimator_params),
        'ExtraTreeClassifier': ExtraTreeClassifier(*estimator_params),
        'GradientBoostingClassifier': GradientBoostingClassifier(*estimator_params),
        'ExtraTreesClassifier': ExtraTreesClassifier(*estimator_params),
        'RandomForestClassifier': RandomForestClassifier(*estimator_params)
    }.get(base_estimator, None)
