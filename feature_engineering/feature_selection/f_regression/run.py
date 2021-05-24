from sklearn.feature_selection import f_regression


def run(x,y, center):
    sk = f_regression(x, y, center=center)
    return {"F": sk[0].tolist(), "pval": sk[1].tolist()}
