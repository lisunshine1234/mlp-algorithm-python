import numpy as np
import run as  r
from sklearn.gaussian_process.kernels import ABCMeta, Matern, ConstantKernel, Exponentiation, ExpSineSquared, Hyperparameter, KernelOperator, \
    NormalizedKernelMixin, PairwiseKernel, RationalQuadratic, StationaryKernelMixin, RBF, CompoundKernel, DotProduct, Product, GenericKernelMixin, WhiteKernel, \
    Kernel, Sum

'''
[id]
112

[name]
GaussianProcessRegressor

[input]
x_train	训练集	训练集标签数据集	二维数组	必须	定数
y_train	测试集	测试集数据集	二维数组	必须	定数
x_test	训练集标签	训练集标签标签	一维数组	必须	定数
y_test	测试集标签	测试集标签	一维数组	必须	定数
kernel	内核	默认为None,指定GP协方差函数的内核。如果传递了None，则默认使用内核'1.0 * RBF(1.0)'。请注意，内核的超参数在拟合过程中已优化,可选字符串	字符串	不必须	定数
alpha	alpha	默认为1e-10,拟合期间将值添加到内核矩阵的对角线。较大的值对应于观测结果中增加的噪声水平。通过确保计算值形成正定矩阵，这也可以防止拟合期间出现潜在的数值问题。如果传递了数组，则该数组必须具有与用于拟合的数据相同的条目数，并且用作与数据点有关的噪声水平。请注意，这等效于添加c = alpha的WhiteKernel。直接允许将噪声级别指定为参数主要是为了方便和与Ridge保持一致,可选数组,浮点数	字符串	不必须	定数
optimizer	optimizer	默认为'fmin_l_bfgs_b',可以是内部支持的用于优化kernel 's parameters, specified by a string, or an externally defined optimizer passed as a callable. Per default, the ' L-BFGS-B ' algorithm from scipy.optimize.minimize is used. If None is passed, the kernel' s参数的优化器之一。可用的内部优化器是：: 'fmin_l_bfgs_b,可选'fmin_l_bfgs_b'	字符串	不必须	定数
n_restarts_optimizer	重新启动次数	默认为0,用于查找内核初始参数的优化程序的重新启动次数，以及从允许的theta值空间中随机抽取的theta采样对数均匀性中剩余的参数(如果有的话)。如果大于0，则所有边界必须是有限的。请注意，n_restarts_optimizer == 0表示执行了一次运行,可选整数	整数	不必须	定数
normalize_y	normalize_y	默认为False,无论目标值y是否被归一化，目标值的均值和方差分别设置为等于0和1。对于使用零均值，单位方差先验的情况，建议使用此方法。注意，在此实现中，在报告GP预测之前，将规范化反转,可选布尔值	布尔值	不必须	定数
copy_X_train	copy_X_train	默认为True,如果为True，则训练数据的永久副本存储在对象中。否则，仅存储对训练数据的引用，如果对数据进行外部修改，则可能导致预测更改,可选布尔值	布尔值	不必须	定数
random_state	随机种子	默认为None,确定用于初始化中心的随机数生成。在多个函数调用之间传递int以获得可重复的结果,可选整数	整数	不必须	定数

[output]
train_predict	预测	训练集预测结果	一维数组(数值)
test_predict	预测	测试集预测结果	一维数组(数值)
train_score	正确率	训练集预测结果的正确率	数字
test_score	正确率	测试集预测结果的正确率	数字
X_train_	X_train_	训练数据的特征向量或其他表示形式(预测也需要)	二维数组
y_train_	y_train_	训练数据中的目标值(预测也需要)	一维数组
L_	L_	'X_train_'中内核的下三角Cholesky分解	二维数组
kernel_	kernel_	用于预测的内核。内核的结构与作为参数传递的内核相同，但具有优化的超参数	字符串
alpha_	alpha	核空间中训练数据点的对偶系数	一维数组
log_marginal_likelihood_value_	对数边际可能性	'self.kernel_.theta'的对数边际可能性	浮点数

[outline]


[describe]
高斯过程回归(GPR)。
该实现基于Rasmussen和Williams提出的高斯机器学习过程算法(GPML)的算法2.1。
除了标准的scikit-learn估计器API外，GaussianProcessRegressor：*允许进行预测而无需事先拟合(基于GP优先级)*提供其他方法sample_y(X)，该方法评估在给定输入下从GPR(优先级或后验)中提取的样本*公开了一个方法log_marginal_likelihood(theta)，该方法可在外部用于其他选择超参数的方式，例如通过马尔可夫链蒙特卡洛。

'''


def main(x_train, y_train, x_test, y_test,
         kernel=None, alpha=1e-10, optimizer="fmin_l_bfgs_b", n_restarts_optimizer=0, normalize_y=False, copy_X_train=True, random_state=None
         ):
    if type(x_train) is str:
        x_train = eval(x_train)
    if type(y_train) is str:
        y_train = eval(y_train)
    if type(x_test) is str:
        x_test = eval(x_test)
    if type(y_test) is str:
        y_test = eval(y_test)
    if type(kernel) is str:
        kernel = eval(kernel)
    if type(alpha) is str:
        alpha = eval(alpha)
    if type(n_restarts_optimizer) is str:
        n_restarts_optimizer = eval(n_restarts_optimizer)
    if type(normalize_y) is str:
        normalize_y = eval(normalize_y)
    if type(copy_X_train) is str:
        copy_X_train = eval(copy_X_train)
    if type(random_state) is str:
        random_state = eval(random_state)

    return r.run(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, kernel=kernel,
                 alpha=alpha,
                 optimizer=optimizer,
                 n_restarts_optimizer=n_restarts_optimizer,
                 normalize_y=normalize_y,
                 copy_X_train=copy_X_train,
                 random_state=random_state)

if __name__ == '__main__':
    import numpy as np
    import json

    array = np.loadtxt('D:\\123_2.csv', delimiter=',')
    array = array[0:20, :]

    y = array[:, -1].tolist()
    x = np.delete(array, -1, axis=1).tolist()
    array = array.tolist()
    back = main(x, y, x, y)

    print(back)
    for i in back:
        print(i + ":" + str(back[i]))

    json.dumps(back)
