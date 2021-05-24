from sklearn.feature_selection import f_classif


def run(x, y):
    sk = f_classif(x, y)
    return {"F": sk[0].tolist(), "pval": sk[1].tolist()}
