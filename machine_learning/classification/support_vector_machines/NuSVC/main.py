import numpy as np
import run as  r

'''
[id]
110

[name]
NuSVR

[input]
x_train	训练集	训练集标签数据集	二维数组	必须	定数
y_train	测试集	测试集数据集	二维数组	必须	定数
x_test	训练集标签	训练集标签标签	一维数组	必须	定数
y_test	测试集标签	测试集标签	一维数组	必须	定数
nu	nu	默认为0.5,训练误差分数的上限和支持向量分数的下限。应该在区间(0，1]中。默认情况下，取0.5,可选浮点数	浮点数	不必须	定数
C	正则化参数	默认为1.0,误差项的惩罚参数C,可选浮点数	浮点数	不必须	定数
kernel	内核	默认为rbf,指定算法中要使用的内核类型。它必须是'linear'，'poly'，'rbf'，'sigmoid'，'precomputed'或可调用项之一。如果没有给出，将使用'rbf'。如果给出了可调用对象，则将其用于预先计算内核矩阵,可选'poly','rbf','sigmoid','precomputed','linear'	字符串	不必须	定数
degree	度	默认为3,多项式内核函数的度数('poly')。被所有其他内核忽略,可选整数	整数	不必须	定数
gamma	gamma	默认为scale,'rbf'，'poly'和'sigmoid'的内核系数。 -如果'gamma=' scale ' (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma, - if ' auto'，则使用1 / n_features,可选浮点数,'auto','scale'	字符串	不必须	定数
coef0	coef0	默认为0.0,内核函数中的独立术语。仅在'poly'和'sigmoid'中有效,可选浮点数	浮点数	不必须	定数
shrinking	启发式方法	默认为True,是否使用缩小的启发式方法,可选布尔值	布尔值	不必须	定数
tol	tol	默认为1e-3,停止标准的公差,可选浮点数	浮点数	不必须	定数
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
intercept_	截距	决策函数中的常量	一维数组

[outline]
Nu支持向量回归。

[describe]
Nu支持向量回归。
与NuSVC相似，对于回归，使用参数nu来控制支持向量的数量。
但是，与NuSVC(其中nu替换C)不同，此处nu替换epsilon-SVR的参数epsilon。
该实现基于libsvm。

'''


def main(x_train, y_train, x_test, y_test,
         nu=0.5, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=1e-3, cache_size=200, class_weight=None,
         verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None
         ):
    if type(x_train) is str:
        x_train = eval(x_train)
    if type(y_train) is str:
        y_train = eval(y_train)
    if type(x_test) is str:
        x_test = eval(x_test)
    if type(y_test) is str:
        y_test = eval(y_test)
    if type(nu) is str:
        nu = eval(nu)
    if type(degree) is str:
        degree = eval(degree)
    if type(coef0) is str:
        coef0 = eval(coef0)
    if type(gamma) is str and gamma != 'scale' and gamma != 'auto':
        gamma = eval(gamma)
    if type(shrinking) is str:
        shrinking = eval(shrinking)
    if type(probability) is str:
        probability = eval(probability)
    if type(tol) is str:
        tol = eval(tol)
    if type(cache_size) is str:
        cache_size = eval(cache_size)
    if type(class_weight) is str and class_weight != 'balanced':
        class_weight = eval(class_weight)
    if type(verbose) is str:
        verbose = eval(verbose)
    if type(max_iter) is str:
        max_iter = eval(max_iter)
    if type(break_ties) is str:
        break_ties = eval(break_ties)
    if type(random_state) is str:
        random_state = eval(random_state)

    return r.run(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, nu=nu,
                 kernel=kernel,
                 degree=degree,
                 gamma=gamma,
                 coef0=coef0,
                 shrinking=shrinking,
                 probability=probability,
                 tol=tol,
                 cache_size=cache_size,
                 class_weight=class_weight,
                 verbose=verbose,
                 max_iter=max_iter,
                 decision_function_shape=decision_function_shape,
                 break_ties=break_ties,
                 random_state=random_state)




if __name__ == '__main__':
    import numpy as np
    import json

    array = np.loadtxt('D:\\123_2.csv', delimiter=',')
    array = array[0:20, :]

    y = array[:, -1].tolist()
    x = np.delete(array, -1, axis=1).tolist()
    array = array.tolist()
    back = main(x, y, x, y,nu=0.1)

    print(back)
    for i in back:
        print(i + ":" + str(back[i]))

    json.dumps(back)
