from sklearn.svm import LinearSVC, LinearSVR, NuSVC, NuSVR, OneClassSVM, SVC, SVR

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def run(x_train, y_train, x_test, y_test,
        nu, C, kernel, degree, gamma, coef0, shrinking, tol, cache_size, verbose, max_iter
        ):
    reg = NuSVR(nu=nu,
                C=C,
                kernel=kernel,
                degree=degree,
                gamma=gamma,
                coef0=coef0,
                shrinking=shrinking,
                tol=tol,
                cache_size=cache_size,
                verbose=verbose,
                max_iter=max_iter
                ).fit(x_train, y_train)
    coef_ = None
    if kernel == 'linear':
        coef_ = reg.coef_.tolist()
    return {'train_predict': reg.predict(x_train).tolist(),
            'test_predict': reg.predict(x_test).tolist(),
            'train_score': reg.score(x_train, y_train),
            'test_score': reg.score(x_test, y_test),
            'support_': reg.support_.tolist(),
            'support_vectors_': reg.support_vectors_.tolist(),
            'dual_coef_': reg.dual_coef_.tolist(),
            'coef_': coef_,
            'intercept_': reg.intercept_.tolist()}
