import numpy as np
import run as  r

'''
[id]
108

[name]
SGDRegressor

[input]
x_train	训练集	训练集标签数据集	二维数组	必须	定数
y_train	测试集	测试集数据集	二维数组	必须	定数
x_test	训练集标签	训练集标签标签	一维数组	必须	定数
y_test	测试集标签	测试集标签	一维数组	必须	定数
loss	损失函数	默认为'hinge',要使用的损失函数。默认为'hinge'，这将提供线性SVM。可能的选项是“hinge”，“log”，“modified_huber”，“squared_hinge”，“perceptron”或回归损失：“squared_loss”，“huber”，“epsilon_insensitive”或“squared_epsilon_insensitive”。对数损失使逻辑回归成为概率分类器。'modified_huber'是另一个平滑的损失，它使异常值和概率估计具有容忍度。“squared_hinge”就像hinge一样，但是被二次惩罚。“感知器”是感知器算法使用的线性损耗。其他损失是为回归而设计的，但也可用于分类,可选字符串	字符串	不必须	定数
penalty	惩罚规范	默认为'l2',要使用的惩罚(又称正则化术语)。默认值为“l2”，这是线性SVM模型的标准正则化器。“l1”和“elasticnet”可能会给模型带来稀疏性(特征选择)，而“l2”是无法实现的,可选l2,l1,elasticnet	字符串	不必须	定数
alpha	alpha	默认为0.0001,与正则项相乘的常数。值越高，正则化越强。当设置为“learning_rate”设置为“optimal”时，也用于计算学习率,可选浮点数	浮点数	不必须	定数
l1_ratio	l1_ratio	默认为0.15,0<=l1_ratio<=1的ElasticNet混合参数。l1_ratio=0对应于L2惩罚，l1_ratio=1到L1。仅在“penalty”为“elasticnet”时使用,可选浮点数	浮点数	不必须	定数
fit_intercept	计算截距	默认为True,是否应该估计截距。如果为False，则假定数据已经居中,可选整数,布尔值	字符串	不必须	定数
max_iter	最大迭代次数	默认为1000,通过训练数据的最大次数(又称历元)。它只会影响'fit'方法的行为，而不会影响：meth：'partial_fit'方法的行为,可选整数	整数	不必须	定数
tol	tol	默认为1e-3,停止标准。如果不是None，则连续(n>_n_iter_no_change_epoch)的训练将在(loss>best_loss-tol)时停止,可选浮点数	浮点数	不必须	定数
shuffle	打乱	默认为True,在每个时期之后是否应重新整理训练数据,可选布尔值	布尔值	不必须	定数
verbose	详细程度	默认为0,详细程度,可选整数	整数	不必须	定数
epsilon	epsilon	默认为0.1,ε对ε不敏感的损失函数；仅当“loss”为“huber”，“对epsilon_insensitive”或“squared_epsilon_insensitive”时。对于“huber”，确定使预测正确正确变得不那么重要的阈值。对于不敏感epsilon的情况，如果当前预测和正确标签之间的任何差异小于此阈值，则将忽略它们,可选浮点数	浮点数	不必须	定数
n_jobs	CPU数量	默认为None,用于执行OVA(对于多类问题，为1对所有)的CPU数量。除非在：obj：'joblib.parallel_backend'上下文中，否则“None'表示1。更多细节,可选整数	整数	不必须	定数
random_state	随机状态	默认为None,当'shuffle'设置为'True'时用于混洗数据。为多个函数调用传递可重复输出的int值,可选整数	整数	不必须	定数
learning_rate	学习率	默认为'optimal',学习率时间表：-'constant'：'eta=eta0'；-'optimal'：'eta=1.0/(alpha*(t+t0))'其中t0由LeonBottou提出的启发式方法选择。-'invscaling'：'eta=eta0/pow(t，power_t)'；-'adaptive'：eta=eta0，只要训练持续减少即可。每次n_iter_no_change个连续的纪元未能将训练损失减少一倍，或者如果Early_stopping为True，则无法增加鉴定分数一倍，则当前学习率除以5,可选字符串	字符串	不必须	定数
eta0	初始学习率	默认为0.0,“constant”，“invscaling”或“adaptive”进度表的初始学习率。默认值为0.0，因为默认计划“optimal”未使用eta0,可选双精度浮点数	双精度浮点数	不必须	定数
power_t	反比例学习率指数	默认为0.5,反比例学习率的指数[默认0.5],可选双精度浮点数	双精度浮点数	不必须	定数
early_stopping	停止终止训练	默认为False,当验证分数没有提高时是否使用提前停止来终止训练。如果设置为True，它将自动将训练数据的分层部分留作验证，并在'score'方法返回的验证分数没有连续n_iter_no_change个时期至少提高tol时终止训练,可选布尔值	布尔值	不必须	定数
validation_fraction	验证分数	默认为0.1,预留的训练数据比例作为早期停止的验证集。必须在0到1之间。仅在“early_stopping”为True时使用,可选浮点数	浮点数	不必须	定数
n_iter_no_change	迭代不变次数	默认为5,迭代次数，没有改善，请提前停止,可选整数	整数	不必须	定数
class_weight	类别权重	默认为None,预设为class_weight-fit参数。与类关联的权重。如果未给出，则所有类都应具有权重一。“balanced”模式使用y的值自动将权重与输入数据中的类频率成反比地调整为“n_samples/(n_classes*np.bincount(y))”,可选字典，字符串	字符串	不必须	定数
warm_start	warm_start	默认为False,设置为True时，请重用上一个调用的解决方案以适合初始化，否则，只需擦除以前的解决方案即可。当warm_start为True时，重复调用fit或partial_fit可能会导致解决方案与一次调用fit时产生不同的解决方案，这是因为数据的混合方式。如果使用动态学习率，则根据已经看到的样本数量调整学习率。调用'fit'将重置此计数器，而'partial_fit'将导致增加现有计数器,可选布尔值	布尔值	不必须	定数
average	平均值	默认为False,设置为True时，将计算所有更新的平均SGD权重并将结果存储在'coef_'属性中。如果将int设置为大于1，则一旦看到的样本总数达到“average”，就会开始平均。因此，'average=10'将在看到10个样本后开始平均,可选整数,布尔值	字符串	不必须	定数

[output]
train_predict	预测	训练集预测结果	一维数组(数值)
test_predict	预测	测试集预测结果	一维数组(数值)
train_score	正确率	训练集预测结果的正确率	数字
test_score	正确率	测试集预测结果的正确率	数字
coef_	参数向量	分配给特征的权重	二维数组
intercept_	截距	决策函数中的常量	整数
n_iter_	迭代次数	达到停止标准之前的实际迭代次数。对于多类拟合，它是每个二进制拟合的最大值	整数
classes_	classes_		一维数组
t_	权重更新次数	训练期间进行的体重更新次数。与(n_iter_*n_samples)相同	整数

[outline]
具有SGD训练的线性分类器。

[describe]
具有SGD训练的线性分类器。
该估计器实现具有随机性的正则化线性模型梯度下降(SGD)学习：
估计损失的梯度一次采样每个样本，并通过降低力量计划(又称学习率)。
SGD允许小批量(partial_fit)方法(在线/核心外)学习。
为了使用默认学习率计划获得最佳结果，数据应均值和单位方差为零。
此实现适用于表示为密集或稀疏数组的数据特征的浮点值。
它适合的模型可以是由损失参数控制；默认情况下，它适合线性支撑向量机(SVM)。
正则化器是对缩小模型的损失函数的惩罚使用平方欧几里德范数向零向量的参数L2或绝对范数L1或两者的组合(弹性网)。
如果由于正则化，参数更新越过0.0值，更新被截断为0.0以允许学习稀疏模型并实现在线特征选择

'''


def main(x_train, y_train, x_test, y_test,
         loss="hinge", penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=1e-3,
         shuffle=True, verbose=0, epsilon=0.1, n_jobs=None, random_state=None, learning_rate="optimal",
         eta0=0.0, power_t=0.5, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None,
         warm_start=False, average=False
         ):
    if type(x_train) is str:
        x_train = eval(x_train)
    if type(y_train) is str:
        y_train = eval(y_train)
    if type(x_test) is str:
        x_test = eval(x_test)
    if type(y_test) is str:
        y_test = eval(y_test)
    if type(alpha) is str:
        alpha = eval(alpha)
    if type(l1_ratio) is str:
        l1_ratio = eval(l1_ratio)
    if type(fit_intercept) is str:
        fit_intercept = eval(fit_intercept)
    if type(max_iter) is str:
        max_iter = eval(max_iter)
    if type(tol) is str:
        tol = eval(tol)
    if type(shuffle) is str:
        shuffle = eval(shuffle)
    if type(verbose) is str:
        verbose = eval(verbose)
    if type(epsilon) is str:
        epsilon = eval(epsilon)
    if type(n_jobs) is str:
        n_jobs = eval(n_jobs)
    if type(random_state) is str:
        random_state = eval(random_state)
    if type(eta0) is str:
        eta0 = eval(eta0)
    if type(power_t) is str:
        power_t = eval(power_t)
    if type(early_stopping) is str:
        early_stopping = eval(early_stopping)
    if type(validation_fraction) is str:
        validation_fraction = eval(validation_fraction)
    if type(n_iter_no_change) is str:
        n_iter_no_change = eval(n_iter_no_change)
    if type(class_weight) is str and class_weight != 'balanced':
        class_weight = eval(class_weight)
    if type(warm_start) is str:
        warm_start = eval(warm_start)
    if type(average) is str:
        average = eval(average)

    return r.run(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,

                 loss=loss,
                 penalty=penalty,
                 alpha=alpha,
                 l1_ratio=l1_ratio,
                 fit_intercept=fit_intercept,
                 max_iter=max_iter,
                 tol=tol,
                 shuffle=shuffle,
                 verbose=verbose,
                 epsilon=epsilon,
                 n_jobs=n_jobs,
                 random_state=random_state,
                 learning_rate=learning_rate,
                 eta0=eta0,
                 power_t=power_t,
                 early_stopping=early_stopping,
                 validation_fraction=validation_fraction,
                 n_iter_no_change=n_iter_no_change,
                 class_weight=class_weight,
                 warm_start=warm_start,
                 average=average)


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