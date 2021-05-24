from sklearn.feature_selection import f_classif, chi2, f_regression, SelectFdr

def run(x, y, score_func, alpha):
    estimated = SelectFdr(score_func=get_score_func(score_func), alpha=alpha)
    x_new = estimated.fit_transform(x, y)
    return {"scores_": estimated.scores_.tolist(), "pvalues_": estimated.pvalues_.tolist(), 'transform': x_new.tolist()}


def get_score_func(score_func):
    return {
        'f_classif': f_classif,
        'chi2': chi2,
        'f_regression': f_regression
    }.get(score_func, f_classif)

