from sklearn.feature_selection import chi2


def run(x, y):
    sk = chi2(x, y)
    return {"chi2": sk[0].tolist(), "pval": sk[1].tolist()}
