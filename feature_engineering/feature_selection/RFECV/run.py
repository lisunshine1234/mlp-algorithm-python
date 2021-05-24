from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import RFECV
from sklearn.linear_model import ARDRegression, BayesianRidge, ElasticNet, ElasticNetCV, Lars, \
    LarsCV, Lasso, LassoCV, LassoLars, LassoLarsCV, LassoLarsIC, LinearRegression, OrthogonalMatchingPursuit, \
    PassiveAggressiveClassifier, PassiveAggressiveRegressor, Perceptron, Ridge, \
    RidgeClassifier, RidgeClassifierCV, RidgeCV, SGDClassifier, SGDRegressor, TheilSenRegressor
from sklearn.svm import SVR, SVC, NuSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier


def run(x, y, estimator, step, min_features_to_select, cv, scoring, verbose):
    selector = RFECV(estimator=get_estimator(estimator),
                     step=step,
                     min_features_to_select=min_features_to_select,
                     cv=cv,
                     scoring=scoring,
                     verbose=verbose)
    selector = selector.fit(x, y)

    return {"n_features_": int(selector.n_features_), "support_": selector.support_.tolist(),
            "ranking_": selector.ranking_.tolist()}


def get_estimator(name):
    return {
        'SVR': SVR(kernel="linear"),
        'SVC': SVC(kernel="linear"),
        'NuSVR': NuSVR(kernel="linear"),
        'ARDRegression': ARDRegression(),
        'BayesianRidge': BayesianRidge(),
        'ElasticNet': ElasticNet(),
        'ElasticNetCV': ElasticNetCV(),
        'Lars': Lars(),
        'LarsCV': LarsCV(),
        'Lasso': Lasso(),
        'LassoCV': LassoCV(),
        'LassoLars': LassoLars(),
        'LassoLarsCV': LassoLarsCV(),
        'LassoLarsIC': LassoLarsIC(),
        'LinearRegression': LinearRegression(),
        'OrthogonalMatchingPursuit': OrthogonalMatchingPursuit(),
        'PassiveAggressiveClassifier': PassiveAggressiveClassifier(),
        'PassiveAggressiveRegressor': PassiveAggressiveRegressor(),
        'Perceptron': Perceptron(),
        'Ridge': Ridge(),
        'RidgeClassifier': RidgeClassifier(),
        'RidgeClassifierCV': RidgeClassifierCV(),
        'RidgeCV': RidgeCV(),
        'SGDClassifier': SGDClassifier(),
        'SGDRegressor': SGDRegressor(),
        'TheilSenRegressor': TheilSenRegressor(),
        'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(),
        'DecisionTreeClassifier': DecisionTreeClassifier(),
        'DecisionTreeRegressor': DecisionTreeRegressor(),
        'ExtraTreeClassifier': ExtraTreeClassifier(),
    }.get(name, SVR(kernel="linear"))
