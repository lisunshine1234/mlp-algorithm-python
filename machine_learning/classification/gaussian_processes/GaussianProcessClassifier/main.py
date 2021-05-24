import numpy as np
import run as  r
from sklearn.gaussian_process.kernels import ABCMeta, Matern, ConstantKernel, Exponentiation, ExpSineSquared, Hyperparameter, KernelOperator, \
    NormalizedKernelMixin, PairwiseKernel, RationalQuadratic, StationaryKernelMixin, RBF, CompoundKernel, DotProduct, Product, GenericKernelMixin, WhiteKernel, \
    Kernel, Sum

'''
[id]
95

[name]
GaussianProcessClassifier

[input]
x_train	训练集	训练集标签数据集	二维数组	必须	定数
y_train	测试集	测试集数据集	二维数组	必须	定数
x_test	训练集标签	训练集标签标签	一维数组	必须	定数
y_test	测试集标签	测试集标签	一维数组	必须	定数
kernel	内核	默认为None,指定GP协方差函数的内核。如果传递了None，则默认使用内核'1.0 * RBF(1.0)'。请注意，内核的超参数在拟合过程中已优化,可选字符串	字符串	不必须	定数
optimizer	optimizer	默认为'fmin_l_bfgs_b',可以是内部支持的用于优化内核的参数（由字符串指定）或作为可调用对象传递的外部定义的优化器。默认情况下，使用scipy.optimize.minimize中的“ L-BFGS-B”算法。如果没有通过，则内核的参数的优化器之一。可用的内部优化器是：'fmin_l_bfgs_b,可选'fmin_l_bfgs_b'	字符串	不必须	定数
n_restarts_optimizer	n_restarts_optimizer	默认为0,用于查找内核初始参数的优化程序的重新启动次数，以及从允许的theta值空间中随机抽取的theta采样对数均匀性中剩余的参数(如果有的话)。如果大于0，则所有边界必须是有限的。请注意，n_restarts_optimizer = 0表示执行了一次运行,可选整数	整数	不必须	定数
max_iter_predict	max_iter_predict	默认为100,在预测过程中，牛顿方法中用于逼近后验的最大迭代次数。较小的值将以更差的结果为代价减少计算时间,可选整数,字典	字符串	不必须	定数
warm_start	warm_start	默认为False,如果启用了热启动，则将后验模式的拉普拉斯近似上最后一次牛顿迭代的解用作下一个_posterior_mode()调用的初始化。在相似的<warm_start>'上多次调用_posterior_mode时，这可以加快收敛速度​​,可选布尔值	布尔值	不必须	定数
copy_X_train	copy_X_train	默认为True,如果为True，则训练数据的永久副本存储在对象中。否则，仅存储对训练数据的引用，如果对数据进行外部修改，则可能导致预测更改,可选布尔值	布尔值	不必须	定数
random_state	随机种子	默认为None,确定用于初始化中心的随机数生成。在多个函数调用之间传递int以获得可重复的结果,可选整数	整数	不必须	定数
multi_class	多类别策略	默认为'one_vs_rest',指定如何处理多类分类问题。支持的是'one_vs_rest'和'one_vs_one'。在'one_vs_rest'中，为每个类别装配一个二进制高斯过程分类器，训练该分类器以将该类别与其余分类分开。在'one_vs_one'中，为每对类别安装一个二进制高斯过程分类器，训练该分类器以将这两个类别分开。这些二进制预测变量的预测被组合为多类预测,可选'one_vs_rest','one_vs_one'	字符串	不必须	定数
n_jobs	CPU数量	默认为None,用于计算的作业数。 'None'表示1，除非在：obj：'joblib.parallel_backend'上下文中。更多细节,可选整数	整数	不必须	定数

[output]
train_predict	预测	训练集预测结果	一维数组(数值)
test_predict	预测	测试集预测结果	一维数组(数值)
train_score	正确率	训练集预测结果的正确率	数字
test_score	正确率	测试集预测结果的正确率	数字
kernel_	kernel_	用于预测的内核。在二进制分类的情况下，内核的结构与作为参数传递的内核的结构相同，但具有优化的超参数。在进行多类分类的情况下，将返回一个CompositeKernel，该复合内核由在“一站式”与“休息”分类器中使用的不同内核组成	字符串
log_marginal_likelihood_value_	log_marginal_likelihood_value_	'self.kernel_.theta'的对数边际可能性	浮点数
classes_	类标签	唯一的类标签	一维数组
n_classes_	n_classes_	训练数据中的班级数	整数

[outline]
基于拉普拉斯近似的高斯过程分类(GPC)。

[describe]
基于拉普拉斯近似的高斯过程分类(GPC)。
该实现基于Rasmussen和Williams提出的高斯机器学习过程(GPML)的算法3.1、3.2和5.1。
在内部，拉普拉斯近似用于通过高斯近似非高斯后验。
当前，该实现仅限于使用逻辑链接特征。
对于多类别分类，安装了几个二进制的一对其余分类器。
注意，该类因此未实现真正的多类拉普拉斯近似。

'''


def main(x_train, y_train, x_test, y_test,
         kernel=None, optimizer="fmin_l_bfgs_b", n_restarts_optimizer=0, max_iter_predict=100, warm_start=False, copy_X_train=True, random_state=None,
         multi_class="one_vs_rest", n_jobs=None
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
    if type(n_restarts_optimizer) is str:
        n_restarts_optimizer = eval(n_restarts_optimizer)
    if type(max_iter_predict) is str:
        max_iter_predict = eval(max_iter_predict)
    if type(warm_start) is str:
        warm_start = eval(warm_start)
    if type(copy_X_train) is str:
        copy_X_train = eval(copy_X_train)
    if type(random_state) is str:
        random_state = eval(random_state)
    if type(n_jobs) is str:
        n_jobs = eval(n_jobs)

    return r.run(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, kernel=kernel,
                 optimizer=optimizer,
                 n_restarts_optimizer=n_restarts_optimizer,
                 max_iter_predict=max_iter_predict,
                 warm_start=warm_start,
                 copy_X_train=copy_X_train,
                 random_state=random_state,
                 multi_class=multi_class,
                 n_jobs=n_jobs)


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