from sklearn.feature_selection import f_oneway


def run(*args):
    sk = f_oneway(*args)
    return {"F-value": sk[0].tolist(), "p-value": sk[1].tolist()}
