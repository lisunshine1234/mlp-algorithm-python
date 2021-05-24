from sklearn.linear_model import orthogonal_mp
import numpy as np

def run(x_train, y_train,
        n_nonzero_coefs, tol, precompute, copy_X, return_path, return_n_iter
        ):
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    reg = orthogonal_mp(x_train, y_train, n_nonzero_coefs=n_nonzero_coefs,
                        tol=tol,
                        precompute=precompute,
                        copy_X=copy_X,
                        return_path=return_path,
                        return_n_iter=return_n_iter)

    if return_n_iter:
        n_iters = reg[1]
        coef = reg[0].tolist()
    else:
        n_iters = None
        coef = reg.tolist()
    return {'coef': coef,
            'n_iters': n_iters}
