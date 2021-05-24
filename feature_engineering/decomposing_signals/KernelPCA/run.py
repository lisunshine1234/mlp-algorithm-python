import numpy as np
from sklearn.decomposition import KernelPCA


def run(array, label_index, n_components,
        kernel,
        gamma,
        degree,
        coef0,
        kernel_params,
        alpha,
        fit_inverse_transform,
        eigen_solver,
        tol,
        max_iter,
        remove_zero_eig,
        random_state,
        copy_X,
        n_jobs):
    x = np.array(array)

    if label_index is not None:
        if label_index < 0:
            label_index = len(array[0]) + label_index

        y = x[:, label_index]
        x = np.delete(array, label_index, axis=1)

    decomposition = KernelPCA(n_components=n_components,
                              kernel=kernel,
                              gamma=gamma,
                              degree=degree,
                              coef0=coef0,
                              kernel_params=kernel_params,
                              alpha=alpha,
                              fit_inverse_transform=fit_inverse_transform,
                              eigen_solver=eigen_solver,
                              tol=tol,
                              max_iter=max_iter,
                              remove_zero_eig=remove_zero_eig,
                              random_state=random_state,
                              copy_X=copy_X,
                              n_jobs=n_jobs)
    x = decomposition.fit_transform(x)
    if label_index is not None:
        x = np.insert(x, len(x[0]), values=y, axis=1)
    dual_coef_ = [[]]
    X_transformed_fit_ = [[]]
    print(fit_inverse_transform)
    if fit_inverse_transform and decomposition.dual_coef_ is not None:
        dual_coef_ = decomposition.dual_coef_.tolist()
        X_transformed_fit_ = decomposition.X_transformed_fit_.tolist()
    return {"array": x.tolist(),
            'lambdas_': decomposition.lambdas_.tolist(),
            'alphas_': decomposition.alphas_.tolist(),
            'dual_coef_': dual_coef_,
            'X_transformed_fit_': X_transformed_fit_,
            'X_fit_': decomposition.X_fit_.tolist()}
