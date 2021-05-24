from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import ARDRegression, BayesianRidge, ElasticNet, ElasticNetCV, HuberRegressor, Lars, LarsCV, \
    Lasso, LassoCV, LassoLars, LassoLarsCV, LassoLarsIC, LinearRegression, OrthogonalMatchingPursuit, \
    OrthogonalMatchingPursuitCV, PassiveAggressiveClassifier, PassiveAggressiveRegressor, Perceptron, \
    Ridge, RidgeClassifier, RidgeClassifierCV, RidgeCV, SGDClassifier, SGDRegressor, TheilSenRegressor
from sklearn.svm import SVR, SVC, NuSVR, OneClassSVM
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor


def run(x, y, estimator, threshold, prefit, norm_order, max_features):
    selector = SelectFromModel(estimator=get_estimator(estimator),
                               threshold=threshold,
                               prefit=prefit,
                               norm_order=norm_order,
                               max_features=max_features)
    x_new = selector.fit_transform(x, y)

    return {"threshold_": selector.threshold_,
            'x_new': x_new.tolist(),
            'support': selector.get_support().tolist()}


def get_estimator(name):
    return {
        'SVR': SVR(kernel="linear"),
        'SVC': SVC(kernel="linear"),
        'NuSVR': NuSVR(kernel="linear"),
        'OneClassSVM': OneClassSVM(kernel="linear"),
        'ARDRegression': ARDRegression(),
        'BayesianRidge': BayesianRidge(),
        'ElasticNet': ElasticNet(),
        'ElasticNetCV': ElasticNetCV(),
        'HuberRegressor': HuberRegressor(),
        'Lars': Lars(),
        'LarsCV': LarsCV(),
        'Lasso': Lasso(),
        'LassoCV': LassoCV(),
        'LassoLars': LassoLars(),
        'LassoLarsCV': LassoLarsCV(),
        'LassoLarsIC': LassoLarsIC(),
        'LinearRegression': LinearRegression(),
        'OrthogonalMatchingPursuit': OrthogonalMatchingPursuit(),
        'OrthogonalMatchingPursuitCV': OrthogonalMatchingPursuitCV(),
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
        'ExtraTreeRegressor': ExtraTreeRegressor()
    }.get(name, SVR(kernel="linear"))
