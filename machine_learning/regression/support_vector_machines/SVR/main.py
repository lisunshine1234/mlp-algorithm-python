import numpy as np
import run as  r

'''
[id]
137

[name]
SVR

[input]
x_train	训练集	训练集标签数据集	二维数组	必须	定数
y_train	测试集	测试集数据集	二维数组	必须	定数
x_test	训练集标签	训练集标签标签	一维数组	必须	定数
y_test	测试集标签	测试集标签	一维数组	必须	定数
kernel	内核	默认为rbf,指定算法中要使用的内核类型。它必须是'linear'，'poly'，'rbf'，'sigmoid'，'precomputed'或可调用项之一。如果没有给出，将使用'rbf'。如果给出了可调用对象，则将其用于预先计算内核矩阵,可选'linear','sigmoid','precomputed','poly','rbf'	字符串	不必须	定数
degree	度	默认为3,多项式内核函数的度数('poly')。被所有其他内核忽略,可选整数	整数	不必须	定数
gamma	gamma	默认为scale,'rbf'，'poly'和'sigmoid'的内核系数。 -如果'gamma=' scale ' (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma, - if ' auto'，则使用1 / n_features,可选浮点数,'auto','scale'	字符串	不必须	定数
coef0	coef0	默认为0.0,内核函数中的独立术语。仅在'poly'和'sigmoid'中有效,可选浮点数	浮点数	不必须	定数
tol	tol	默认为1e-3,停止标准的公差,可选浮点数	浮点数	不必须	定数
C	正则化参数	默认为1.0,正则化参数。正则化的强度与C成反比。必须严格为正。惩罚是平方的l2惩罚,可选浮点数	浮点数	不必须	定数
epsilon	epsilon	默认为0.1,epsilon-SVR模型中的Epsilon。它指定在训练损失函数中没有罚分的epsilon管，该点与实际值之间的距离为epsilon,可选浮点数	浮点数	不必须	定数
shrinking	启发式方法	默认为True,是否使用缩小的启发式方法。参见：ref：'User Guide <shrinking_svm>',可选布尔值	布尔值	不必须	定数
cache_size	缓存大小	默认为200,指定内核缓存的大小(以MB为单位),可选浮点数	浮点数	不必须	定数
verbose	详细程度	默认为False,启用详细输出。请注意，此设置利用了libsvm中每个进程的运行时设置，如果启用，则可能无法在多线程上下文中正常工作,可选布尔值	布尔值	不必须	定数
max_iter	最大迭代次数	默认为-1,对求解器内的迭代进行硬性限制，或者为-1(无限制),可选整数	整数	不必须	定数

[output]
train_predict	预测	训练集预测结果	一维数组(数值)
test_predict	预测	测试集预测结果	一维数组(数值)
train_score	正确率	训练集预测结果的正确率	数字
test_score	正确率	测试集预测结果的正确率	数字
support_	支持向量指标	支持向量的指标	一维数组
support_vectors_	支持向量	支持向量	二维数组
dual_coef_	权重向量	决策函数中支持向量的系数	二维数组
coef_	参数向量	分配给特征的权重(原始问题的系数)。仅在线性内核的情况下可用。 'coef_'是从'dual_coef_'和'support_vectors_'派生的只读属性	二维数组
fit_status_	fit_status_	如果正确安装，则为0，否则为1(将发出警告)	整数
intercept_	截距	决策函数中的常量	一维数组

[outline]
支持向量回归。

[describe]
Epsilon支持向量回归。
模型中的自由参数是C和epsilon。
该实现基于libsvm。
拟合时间的复杂度是样本数量的两倍以上，这使得很难扩展到具有多个10000个样本的数据集。
对于大型数据集，请考虑改用SGDRegressor，可能在Nystroem变压器之后。

'''


def main(x_train, y_train, x_test, y_test,
         kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=1e-3, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1
         ):
    if type(x_train) is str:
        x_train = eval(x_train)
    if type(y_train) is str:
        y_train = eval(y_train)
    if type(x_test) is str:
        x_test = eval(x_test)
    if type(y_test) is str:
        y_test = eval(y_test)
    if type(degree) is str:
        degree = eval(degree)
    if type(coef0) is str:
        coef0 = eval(coef0)
    if type(tol) is str:
        tol = eval(tol)
    if type(gamma) is str and gamma != 'scale' and gamma != 'auto':
        gamma = eval(gamma)
    if type(C) is str:
        C = eval(C)
    if type(epsilon) is str:
        epsilon = eval(epsilon)
    if type(shrinking) is str:
        shrinking = eval(shrinking)
    if type(cache_size) is str:
        cache_size = eval(cache_size)
    if type(verbose) is str:
        verbose = eval(verbose)
    if type(max_iter) is str:
        max_iter = eval(max_iter)

    return r.run(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, kernel=kernel,
                 degree=degree,
                 gamma=gamma,
                 coef0=coef0,
                 tol=tol,
                 C=C,
                 epsilon=epsilon,
                 shrinking=shrinking,
                 cache_size=cache_size,
                 verbose=verbose,
                 max_iter=max_iter)


if __name__ == '__main__':
    import numpy as np
    import json

    array = np.loadtxt('D:\\123_2.csv', delimiter=',')
    array = array[0:20, :]

    y = array[:, -1].tolist()
    x = np.delete(array, -1, axis=1).tolist()
    array = array.tolist()
    back = main(x, y,x,y)

    print(back)
    for i in back:
        print(i + ":" + str(back[i]))

    json.dumps(back)

