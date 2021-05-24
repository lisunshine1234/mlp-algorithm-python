import numpy as np
import run as r
'''
[id]
93

[name]
KernelPCA

[input]
array 数据集 数据集 二维数组 必须 定数
label_index 标签列号 默认为None,表示所有列参与训练，输入整数代表标签所在的列号，可选整数 数字 不必须 定数
n_components 组件数 默认为None,组件数。如果为None，则将保留所有非零分量,可选整数 整数 不必须 定数
kernel 核 默认为linear,可选linear,poly,rbf,sigmoid,cosine,precomputed 字符串 不必须 定数
gamma gamma 默认为1/n_features,rbf，poly和sigmoid内核的内核系数。被其他内核忽略,可选浮点数 浮点数 不必须 定数
degree 度 默认为3,poly内核的度。被其他内核忽略,可选整数 整数 不必须 定数
coef0 系数0 默认为1,poly和sigmoid内核的内核系数。被其他内核忽略,可选浮点数 浮点数 不必须 定数
kernel_params 内核参数 默认为None,作为可调用对象传递的内核的参数（关键字参数）和值。被其他内核忽略,可选字符串 字符串 不必须 定数
alpha alpha 默认为1.0,岭回归的超参数，用于学习逆变换（当fit_inverse_transform=True时）,可选整数 整数 不必须 定数
fit_inverse_transform 拟合逆变换 默认为False,获得非预计算内核的逆变换。（即学会找到一个点的原像,可选布尔值 布尔值 不必须 定数
eigen_solver 求解器 默认为'auto',选择要使用的本征求解器。如果n_components远小于训练样本的数量，则arpack可能比密集本征求解器更有效,可选auto,dense,arpack 字符串 不必须 定数
tol tol 默认为0,arpack的收敛容限。如果为0，则arpack将选择最佳值,可选浮点数 浮点数 不必须 定数
max_iter 最大迭代次数 默认为None,arpack的最大迭代次数。如果为None，则arpack将选择最佳值,可选整数 整数 不必须 定数
remove_zero_eig 删除零 默认为False,如果为True，则将删除所有具有零特征值的分量，以使输出中的分量数可能小于<n_components（有时由于数字不稳定性甚至为零）。当n_components为None时，将忽略此参数，并删除特征值为零的组件,可选布尔值 布尔值 不必须 定数
random_state 随机状态 默认为None,当``eigen_solver``=='arpack'时使用。在多个函数调用之间传递int以获得可重复的结果,可选整数 整数 不必须 定数
copy_X 是否复制 默认为True,如果为True，则模型将输入X复制并存储在X_fit_属性中。如果对X不再做任何更改，则设置`copy_X=False`可以通过存储引用来节省内存,可选布尔值 布尔值 不必须 定数
n_jobs n_jobs 默认为None,要运行的并行作业数。除非在：obj：`joblib.parallel_backend`上下文中，否则“None``表示1。-1表示使用所有处理器,可选整数 整数 不必须 定数


[output]
array 数组 训练之后的数组 二维数组
lambdas_ lambdas 中心核矩阵的特征值以降序排列。如果未设置“n_components”和“remove_zero_eig”，则将存储所有值。 一维数组
alphas_ alphas 中心核矩阵的特征向量。如果未设置`n_components`和`remove_zero_eig`，则存储所有组件 二维数组
dual_coef_ dual_coef 逆变换矩阵。仅在``fit_inverse_transform''为True时可用 二维数组
X_transformed_fit_ X_transformed_fit_ 拟合数据在内核主成分上的投影。仅在``fit_inverse_transform''为True时可用 二维数组
X_fit_ X_fit_ 用于拟合模型的数据。如果`copy_X=False`，则`X_fit_`是参考。此属性用于进行转换的调用 二维数组

[outline]
内核主成分分析（KPCA）。

[describe]
内核主成分分析（KPCA）通过使用内核来减少非线性维数。
'''

def main(array, label_index=None, n_components=None, kernel="linear",
         gamma=None, degree=3, coef0=1, kernel_params=None,
         alpha=1.0, fit_inverse_transform=False, eigen_solver='auto',
         tol=0, max_iter=None, remove_zero_eig=False,
         random_state=None, copy_X=True, n_jobs=None):
    if type(array) is str:
        array = eval(array)
    if type(label_index) is str:
        label_index = eval(label_index)
    if type(n_components) is str:
        n_components = eval(n_components)
    if type(gamma) is str:
        gamma = eval(gamma)
    if type(degree) is str:
        degree = eval(degree)
    if type(coef0) is str:
        coef0 = eval(coef0)
    if type(alpha) is str:
        alpha = eval(alpha)
    if type(fit_inverse_transform) is str:
        fit_inverse_transform = eval(fit_inverse_transform)
    if type(tol) is str:
        tol = eval(tol)
    if type(max_iter) is str:
        max_iter = eval(max_iter)
    if type(remove_zero_eig) is str:
        remove_zero_eig = eval(remove_zero_eig)
    if type(random_state) is str:
        random_state = eval(random_state)
    if type(copy_X) is str:
        copy_X = eval(copy_X)
    if type(n_jobs) is str:
        n_jobs = eval(n_jobs)
    return r.run(array=array,
                 label_index=label_index,
                 n_components=n_components,
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


if __name__ == '__main__':
    import numpy as np
    import json
    array = np.loadtxt('D:\\123_2.csv', delimiter=',')
    y = array[:, -1]
    x = np.delete(array, -1, axis=1)

    x = x.tolist()
    y = y.tolist()
    array = array.tolist()
    print(main(array, -1, n_components=2))
