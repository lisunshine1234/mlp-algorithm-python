from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import ARDRegression, BayesianRidge, ElasticNet, ElasticNetCV, HuberRegressor, Lars, LarsCV, \
    Lasso, LassoCV, LassoLars, LassoLarsCV, LassoLarsIC, LinearRegression, OrthogonalMatchingPursuit, \
    OrthogonalMatchingPursuitCV, PassiveAggressiveClassifier, PassiveAggressiveRegressor, Perceptron, \
    Ridge, RidgeClassifier, RidgeClassifierCV, RidgeCV, SGDClassifier, SGDRegressor, TheilSenRegressor
from sklearn.svm import SVR, SVC, NuSVR, OneClassSVM
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier


def run(x_train, y_train, x_test, y_test,
        loss, estimator_params, learning_rate, n_estimators, subsample, criterion, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_depth,
        min_impurity_decrease, min_impurity_split, init, random_state, max_features, verbose, max_leaf_nodes, warm_start, validation_fraction, n_iter_no_change,
        tol, ccp_alpha
        ):
    if init != 'zero':
        init = get_estimator(init, estimator_params)
    reg = GradientBoostingClassifier(loss=loss,
                                     learning_rate=learning_rate,
                                     n_estimators=n_estimators,
                                     subsample=subsample,
                                     criterion=criterion,
                                     min_samples_split=min_samples_split,
                                     min_samples_leaf=min_samples_leaf,
                                     min_weight_fraction_leaf=min_weight_fraction_leaf,
                                     max_depth=max_depth,
                                     min_impurity_decrease=min_impurity_decrease,
                                     min_impurity_split=min_impurity_split,
                                     init=init,
                                     random_state=random_state,
                                     max_features=max_features,
                                     verbose=verbose,
                                     max_leaf_nodes=max_leaf_nodes,
                                     warm_start=warm_start,
                                     validation_fraction=validation_fraction,
                                     n_iter_no_change=n_iter_no_change,
                                     tol=tol,
                                     ccp_alpha=ccp_alpha).fit(x_train, y_train)
    oob_improvement_ = None
    if subsample < 1.0:
        oob_improvement_ = reg.oob_improvement_.tolist()
    return {'train_predict': reg.predict(x_train).tolist(),
            'test_predict': reg.predict(x_test).tolist(),
            'train_score': reg.score(x_train, y_train),
            'test_score': reg.score(x_test, y_test),
            'n_estimators_': reg.n_estimators_,
            'feature_importances_': reg.feature_importances_.tolist(),
            'oob_improvement_': oob_improvement_,
            'train_score_': reg.train_score_.tolist(),
            'classes_': reg.classes_.tolist(),
            'n_features_': reg.n_features_,
            'n_classes_': reg.n_classes_,
            'max_features_': reg.max_features_
            }


def get_estimator(base_estimator, estimator_params):
    if base_estimator is None:
        return base_estimator
    base_estimator.replace("(", "").replace(")", "")
    if estimator_params is None:
        estimator_params = {}
    return {
        'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(*estimator_params),
        'DecisionTreeClassifier': DecisionTreeClassifier(*estimator_params),
        'ExtraTreeClassifier': ExtraTreeClassifier(*estimator_params),
        'GradientBoostingClassifier': GradientBoostingClassifier(*estimator_params),
        'ExtraTreesClassifier': ExtraTreesClassifier(*estimator_params),
        'RandomForestClassifier': RandomForestClassifier(*estimator_params)
    }.get(base_estimator, None)
