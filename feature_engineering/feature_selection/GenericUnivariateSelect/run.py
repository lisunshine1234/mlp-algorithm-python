from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.feature_selection import f_classif, mutual_info_classif, chi2, f_regression, mutual_info_regression, \
    SelectPercentile, SelectKBest, SelectFpr, SelectFdr, SelectFwe


def run(x, y, score_func, mode, param):
    sk = GenericUnivariateSelect(score_func=get_score_func(score_func), mode=mode, param=param)
    x = sk.fit_transform(x, y)
    return {"pvalues_": sk.pvalues_.tolist(), "scores_": sk.scores_.tolist()}


def get_score_func(score_func):
    return {
        'f_classif': f_classif,
        'mutual_info_classif': mutual_info_classif,
        'chi2': chi2,
        'f_regression': f_regression,
        'mutual_info_regression': mutual_info_regression,
        'SelectPercentile': SelectPercentile,
        'SelectKBest': SelectKBest,
        'SelectFpr': SelectFpr,
        'SelectFdr': SelectFdr,
        'SelectFwe': SelectFwe,
    }.get(score_func, f_classif)
