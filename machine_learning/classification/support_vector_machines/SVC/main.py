import numpy as np
import run as  r
import json
'''
[id]
111

[name]
SVC

[input]
x_train	训练集	训练集标签数据集	二维数组	必须	定数
y_train	测试集	测试集数据集	二维数组	必须	定数
x_test	训练集标签	训练集标签标签	一维数组	必须	定数
y_test	测试集标签	测试集标签	一维数组	必须	定数
C	C	默认为1.0,正则化参数。正则化的强度与C成反比。必须严格为正。惩罚是平方的l2惩罚,可选浮点数	浮点数	不必须	定数
kernel	内核	默认为rbf,指定算法中使用的内核类型，它必须是'linear'，'poly'，'rbf'，'sigmoid'，'precomputed'，'rbf' ora之一，如果没有给出，将使用'(n_samples, n_samples)'。如果给出了可调用对象，则用于从数据矩阵中预先计算内核矩阵。该矩阵应该是形状为$的数组,可选'linear','poly','rbf','sigmoid','precomputed','rbf'	字符串	不必须	定数
degree	度	默认为3,多项式内核函数('poly')的度，被所有其他内核忽略,可选整数	整数	不必须	定数
gamma	gamma	默认为scale,'rbf'，'poly'和'sigmoid'的内核系数-如果'gamma=' scale ' (default) is passed then it uses1 / (n_features * X.var()) as value of gamma,- if ' auto'使用1 / n_features,可选浮点数,'scale','auto','scale'	字符串	不必须	定数
coef0	coef0	默认为0.0,内核函数中的独立项，仅在'poly'和'sigmoid'中有效,可选浮点数	浮点数	不必须	定数
shrinking	启发式方法	默认为True,是否使用缩小的启发式方法,可选布尔值	布尔值	不必须	定数
probability	概率估计	默认为False,是否启用概率估计。必须在调用'fit'之前启用此特征，因为它在内部使用5折交叉验证，因此会降低该方法的速度，并且'predict_proba'可能与,可选布尔值	布尔值	不必须	定数
tol	tol	默认为1e-3,停止标准的公差,可选浮点数	浮点数	不必须	定数
cache_size	cache_size	默认为200,指定内核缓存的大小(以MB为单位),可选浮点数	浮点数	不必须	定数
class_weight	类别权重	默认为None,对于SVC，将类i的参数C设置为class_weight [i] * C。如果未给出，则所有类都应具有一个权重。'balanced'模式使用y的值自动将权重与输入数据中的类频率成反比地调整为'n_samples / (n_classes * np.bincount(y)),可选'balanced'	字符串	不必须	定数
verbose	详细程度	默认为False,启用详细输出。请注意，此设置利用libsvm中的按进程运行时设置，如果启用了该设置，则在多线程上下文中可能无法正常工作,可选布尔值	布尔值	不必须	定数
max_iter	最大迭代次数	默认为-1,对求解器内的迭代进行硬性限制，或者为-1(无限制),可选整数	整数	不必须	定数
decision_function_shape	决策函数形状	默认为ovr,是否要返回形状(n_samples，n_classes)的一对一('ovr')决策函数作为所有其他分类器，还是返回形状为(n_samples，n_classes *(n_classes)的libsvm的原始一对一('ovo')决策函数-1)/ 2)。但是，始终将one-vs-one('ovo')用作多类策略。对于二进制分类，将忽略该参数,可选'ovo','ovr'	字符串	不必须	定数
break_ties	break_ties	默认为False,如果为true，则'decision_function_shape=' ovr ', and number of classes > 2,' predict ' 将根据以下项的置信度值打破联系:' decision_function';否则，将返回tieedclass中的第一类。请注意，与简单的预测相比，打破平局要付出相当高的计算成本,可选布尔值	布尔值	不必须	定数
random_state	随机种子	默认为None,控制伪随机数的生成，以对数据进行混洗以进行概率估计。当'probability'为False时被忽略，在多个函数调用之间传递一个可重复输出的int,可选整数	整数	不必须	定数

[output]
train_predict	预测	训练集预测结果	一维数组(数值)
test_predict	预测	测试集预测结果	一维数组(数值)
train_score	正确率	训练集预测结果的正确率	数字
test_score	正确率	测试集预测结果的正确率	数字
support_	支持向量指标	支持向量的指标	一维数组
support_vectors_	支持向量	支持向量	二维数组
n_support_	类支持向量数量	每个类的支持向量数量	一维数组
dual_coef_	权重向量	决策函数中支持向量的对偶系数(参见：ref：'sgd_mathematical_formulation')乘以其目标;对于多类而言，所有1-vs-1分类器的系数。在多类情况下，系数的布局有些微不足道。有关详细信息，请参见：ref：'multi-class section of the User Guide<svm_multi_class>'	二维数组
coef_	参数向量	分配给特征的权重(原始问题中的系数)。这仅在线性内核的情况下可用。'coef_'是从'dual_coef_'和'support_vectors_'派生的只读属性	一维数组
intercept_	截距	决策函数中的常量	一维数组
fit_status_	fit_status_	如果正确安装，则为0，否则为1(将发出警告)	整数
classes_	类标签	类标签	一维数组
probA_	probA_		一维数组
probB_	probB_	如果'probability=True'，则对应于在Platt缩放中学习的参数，以根据决策值产生概率估计。如果'probability=False'，则从数据集中学习's an empty array. Platt scaling uses thelogistic function' 1 / /(1 + exp(decision_value * probA_ + probB _))'where ' probA_ ' and ' probB_'[2] _。	一维数组
class_weight_	类别权重	每个类的参数C的乘数，基于'class_weight'参数计算	一维数组
shape_fit_	训练向量数组维数	训练向量'X'的数组维数	整数

[outline]
支持向量分类,该实现基于libsvm

[describe]
C支持向量分类。该实现基于libsvm。
拟合时间至少与样本数量成平方比例，超过成千上万的样本可能不切实际。
对于大型数据集，请考虑改用SGDClassifier，可能在Nystroem转换器之后。
多类支持根据一对一方案进行处理。
'''


def main(x_train, y_train, x_test, y_test,
         C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=1e-3, cache_size=200, class_weight=None, verbose=False,
         max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None
         ):
    if type(x_train) is str:
        x_train = eval(x_train)
    if type(y_train) is str:
        y_train = eval(y_train)
    if type(x_test) is str:
        x_test = eval(x_test)
    if type(y_test) is str:
        y_test = eval(y_test)
    if type(C) is str:
        C = eval(C)
    if type(degree) is str:
        degree = eval(degree)
    if type(coef0) is str:
        coef0 = eval(coef0)
    if type(shrinking) is str:
        shrinking = eval(shrinking)
    if type(probability) is str:
        probability = eval(probability)
    if type(tol) is str:
        tol = eval(tol)
    if type(cache_size) is str:
        cache_size = eval(cache_size)
    if type(gamma) is str and gamma != 'scale' and gamma != 'auto':
        gamma = eval(gamma)
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

    return r.run(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, C=C,
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
    back = main(x, y, x, y)

    print(back)
    for i in back:
        print(i + ":" + str(back[i]))

    json.dumps(back)
