import numpy as np
import run as  r

'''
[id]
109

[name]
LinearSVC

[input]
x_train	训练集	训练集标签数据集	二维数组	必须	定数
y_train	测试集	测试集数据集	二维数组	必须	定数
x_test	训练集标签	训练集标签标签	一维数组	必须	定数
y_test	测试集标签	测试集标签	一维数组	必须	定数
penalty	惩罚	默认为l2,指定惩罚中使用的规范。罚款是SVC中使用的标准。 'l2'导致稀疏的'coef_' vector,可选'l2','l1'	字符串	不必须	定数
loss	损失函数	默认为squared_hinge,指定损失函数。 'hinge'是标准SVM损耗(例如由SVC类使用)，而'squared_hinge'是hinge损耗的平方,可选'hinge','squared_hinge'	字符串	不必须	定数
dual	dual	默认为True,选择算法来解决对偶优化或原始优化问题。当n_samples> n_features时，首选dual = False,可选布尔值	布尔值	不必须	定数
tol	tol	默认为1e-4,停止标准的公差,可选浮点数	浮点数	不必须	定数
C	正则化参数	默认为1.0,正则化参数。正则化的强度与C成反比。必须严格为正,可选浮点数	浮点数	不必须	定数
multi_class	多类别策略	默认为ovr,如果'crammer_singer'包含两个以上的类，则确定多类别策略。'ovr'将n_class训练为一个-vs-rest分类器，而'y'优化所有类的联合目标。如果选择“ crammer_singer”，则选项损失，惩罚和对偶将被忽略,可选'crammer_singer','ovr'	字符串	不必须	定数
fit_intercept	计算截距	默认为True,是否计算此模型的截距。如果将其设置为false，则在计算中将不使用截距(即，数据应已居中),可选整数,布尔值	字符串	不必须	定数
intercept_scaling	截距缩放	默认为1,当self.fit_intercept为True时，实例向量x变为'[x, self.intercept_scaling]'，即。实例向量后会附加一个常数值等于intercept_scaling的'synthetic'特征。截距将变为intercept_scaling,与其他所有特征一样，合成特征权重也要经过l1 / l2正则化。要减轻正则化对合成特征权重(并因此对截距的影响)，必须增加intercept_scaling,可选整数,浮点数	字符串	不必须	定数
class_weight	类别权重	默认为None,将类i的参数C设置为'class_weight[i]*C' forSVC。如果未给出，则所有类都应该具有一个权重。'n_samples / (n_classes * np.bincount(y))'模式使用y的值自动将权重与输入数据中的类频率成反比地调整为'balanced',可选'balanced'	字符串	不必须	定数
verbose	详细程度	默认为0,启用详细输出。请注意，此设置利用liblinear中的按进程运行时设置，如果启用了该设置，则在多线程上下文中可能无法正常工作,可选整数	整数	不必须	定数
random_state	随机种子	默认为None,控制伪随机数生成，以便为双坐标下降(如果'LinearSVC')对数据进行混洗。当'random_state'的基础实现不是随机的并且'dual=True'对结果没有影响时，请在多个函数调用之间传递一个可重复输出的int,可选整数	整数	不必须	定数
max_iter	最大迭代次数	默认为1000,要运行的最大迭代次数,可选整数	整数	不必须	定数

[output]
train_predict	预测	训练集预测结果	一维数组(数值)
test_predict	预测	测试集预测结果	一维数组(数值)
train_score	正确率	训练集预测结果的正确率	数字
test_score	正确率	测试集预测结果的正确率	数字
coef_	参数向量	分配给特征的权重(原始问题中的系数)。这仅在线性内核的情况下可用。'raw_coef_'是从'coef_'派生的只读属性，其遵循liblinear的内部内存布局	二维数组
intercept_	截距	决策函数中的常量	一维数组
classes_	类标签	唯一的类标签	一维数组
n_iter_	迭代次数	所有类的最大迭代次数	整数

[outline]
线性支持向量分类，类似于带有参数kernel ='linear'的SVC，但在实现方面 liblinear而不是libsvm，

[describe]
线性支持向量回归。
类似于带有参数kernel ='linear'的SVR，但它是根据liblinear而不是libsvm来实现的，
因此它在选择罚分和损失函数时具有更大的灵活性，应更好地扩展到大量样本。此类同时支持密集输入和稀疏输入。
'''


def main(x_train, y_train, x_test, y_test,
         penalty='l2', loss='squared_hinge', dual=True, tol=1e-4, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None,
         verbose=0, random_state=None, max_iter=1000
         ):
    if type(x_train) is str:
        x_train = eval(x_train)
    if type(y_train) is str:
        y_train = eval(y_train)
    if type(x_test) is str:
        x_test = eval(x_test)
    if type(y_test) is str:
        y_test = eval(y_test)
    if type(dual) is str:
        dual = eval(dual)
    if type(tol) is str:
        tol = eval(tol)
    if type(C) is str:
        C = eval(C)
    if type(fit_intercept) is str:
        fit_intercept = eval(fit_intercept)
    if type(intercept_scaling) is str:
        intercept_scaling = eval(intercept_scaling)
    if type(class_weight) is str and class_weight != 'balanced':
        class_weight = eval(class_weight)
    if type(verbose) is str:
        verbose = eval(verbose)
    if type(random_state) is str:
        random_state = eval(random_state)
    if type(max_iter) is str:
        max_iter = eval(max_iter)

    return r.run(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, penalty=penalty,
                 loss=loss,
                 dual=dual,
                 tol=tol,
                 C=C,
                 multi_class=multi_class,
                 fit_intercept=fit_intercept,
                 intercept_scaling=intercept_scaling,
                 class_weight=class_weight,
                 verbose=verbose,
                 random_state=random_state,
                 max_iter=max_iter)

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
