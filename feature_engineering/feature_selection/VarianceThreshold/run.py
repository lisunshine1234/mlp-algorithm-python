from sklearn.feature_selection import VarianceThreshold


def run(x, y, threshold):
    selector = VarianceThreshold(threshold=threshold)
    x_new = selector.fit_transform(x, y)
    return {"scores_": selector.variances_.tolist(), 'transform': x_new.tolist()}
