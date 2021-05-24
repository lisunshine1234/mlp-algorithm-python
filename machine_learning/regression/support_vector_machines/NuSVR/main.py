import numpy as np
import run as  r

'''
[id]
136

[name]
NuSVR

[input]
x_train	训练集	训练集标签数据集	二维数组	必须	定数
y_train	测试集	测试集数据集	二维数组	必须	定数
x_test	训练集标签	训练集标签标签	一维数组	必须	定数
y_test	测试集标签	测试集标签	一维数组	必须	定数
nu	nu	默认为0.5,边际误差分数的上限和支持向量分数的下限。应该在间隔(0，1]中,可选浮点数	浮点数	不必须	定数
kernel	内核	默认为rbf,指定算法中要使用的内核类型。它必须是'linear'，'poly'，'rbf'，'sigmoid'，'precomputed'或可调用项之一。如果没有给出，将使用'rbf'。如果给出了可调用对象，则将其用于预先计算内核矩阵,可选'sigmoid','poly','linear','precomputed','rbf'	字符串	不必须	定数
degree	度	默认为3,多项式内核函数的度数('poly')。被所有其他内核忽略,可选整数	整数	不必须	定数
gamma	gamma	默认为scale,'rbf'，'poly'和'sigmoid'的内核系数。 -如果'gamma=' scale ' (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma, - if ' auto'，则使用1 / n_features,可选浮点数,'scale','auto'	字符串	不必须	定数
coef0	coef0	默认为0.0,内核函数中的独立术语。仅在'poly'和'sigmoid'中有效,可选浮点数	浮点数	不必须	定数
shrinking	启发式方法	默认为True,是否使用缩小的启发式方法,可选布尔值	布尔值	不必须	定数
probability	概率估计	默认为False,是否启用概率估计。必须在调用'fit'之前启用此特征，因为它在内部使用5倍交叉验证，并且'predict_proba'可能与'fit'不一致，因此会减慢该方法的速度,可选布尔值	布尔值	不必须	定数
tol	tol	默认为1e-3,停止标准的公差,可选浮点数	浮点数	不必须	定数
cache_size	缓存大小	默认为200,指定内核缓存的大小(以MB为单位),可选浮点数	浮点数	不必须	定数
class_weight	类别权重	默认为None,对于SVC，将类i的参数C设置为class_weight [i] * C。如果未给出，则所有类都应具有权重一。 'balanced'模式使用y的值自动将与班级频率成反比的权重调整为'n_samples / (n_classes * np.bincount(y)),可选'balanced'	字符串	不必须	定数
verbose	详细程度	默认为False,启用详细输出。请注意，此设置利用了libsvm中每个进程的运行时设置，如果启用，则可能无法在多线程上下文中正常工作,可选布尔值	布尔值	不必须	定数
max_iter	最大迭代次数	默认为-1,对求解器内的迭代进行硬性限制，或者为-1(无限制),可选整数	整数	不必须	定数
decision_function_shape	决策函数形状	默认为ovr,是否返回形状(n_samples，n_classes)的一对一('ovr')决策函数作为所有其他分类器，还是返回形状为(n_samples，n_classes *( n_classes-1)/ 2)。但是，始终将一对多('ovo')用作多类别策略。对于二进制分类，将忽略该参数,可选'ovo','ovr'	字符串	不必须	定数
break_ties	break_ties	默认为False,如果为true，则'decision_function_shape=' ovr ', classes > 2,'predict'将根据'decision_function'的置信度值断开联系;否则，返回绑定类中的第一类,可选布尔值	布尔值	不必须	定数
random_state	随机种子	默认为None,控制伪随机数生成，以对数据进行混洗以进行概率估计。当'probability'为False时被忽略。为多个函数调用传递可重复输出的int值,可选整数	整数	不必须	定数

[output]
train_predict	预测	训练集预测结果	一维数组(数值)
test_predict	预测	测试集预测结果	一维数组(数值)
train_score	正确率	训练集预测结果的正确率	数字
test_score	正确率	测试集预测结果的正确率	数字
support_	支持向量指标	支持向量的指标	一维数组
support_vectors_	支持向量	支持向量	二维数组
n_support_	类支持向量数量	每个类的支持向量数量	一维数组
dual_coef_	权重向量	决策函数中支持向量的对偶系数(请参见：ref：'sgd_mathematical_formulation')乘以其目标。对于多类别，所有1-vs-1分类器的系数。在多类情况下，系数的布局有些微不足道。有关详细信息，请参见：ref：'multi-class section of the User Guide <svm_multi_class>'	二维数组
coef_	参数向量	分配给特征的权重(原始问题的系数)。仅在线性内核的情况下可用。 'coef_'是从'dual_coef_'和'support_vectors_'派生的只读属性	一维数组
intercept_	截距	决策函数中的常量	一维数组
classes_	类标签	唯一的类标签	一维数组
fit_status_	fit_status_	如果正确拟合，则为0；如果算法未收敛，则为1	整数
probA_	probA_	无描述信息	一维数组
probB_	probB_	如果为'probability=True'，则它对应于在Platt缩放中学习的参数，以根据决策值产生概率估计。如果是'probability=False'，则从数据集[2] _中获悉's an empty array. Platt scaling uses the logistic function ' 1 / /(1 + exp(decision_value * probA_ + probB _))' where ' probA_ ' and ' probB_'。	一维数组
class_weight_	类别权重	每个类的参数C的乘数。根据'class_weight'参数进行计算	一维数组
shape_fit_	shape_fit_	训练向量'X'的数组维数	整数
[outline]
Nu支持向量分类
与SVC相似，但使用参数来控制支持向量的数量

[describe]
Nu支持向量分类与SVC相似，但使用参数来控制支持向量的数量
该实现基于libsvm

'''


def main(x_train, y_train, x_test, y_test,
         nu=0.5, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, tol=1e-3, cache_size=200, verbose=False, max_iter=-1
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
    if type(C) is str:
        C = eval(C)
    if type(degree) is str:
        degree = eval(degree)
    if type(gamma) is str and gamma != 'scale' and gamma != 'auto':
        gamma = eval(gamma)
    if type(coef0) is str:
        coef0 = eval(coef0)
    if type(shrinking) is str:
        shrinking = eval(shrinking)
    if type(tol) is str:
        tol = eval(tol)
    if type(cache_size) is str:
        cache_size = eval(cache_size)
    if type(verbose) is str:
        verbose = eval(verbose)
    if type(max_iter) is str:
        max_iter = eval(max_iter)

    return r.run(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, nu=nu,
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
                 )


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

