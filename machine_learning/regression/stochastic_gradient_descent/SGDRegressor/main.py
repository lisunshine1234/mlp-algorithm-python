import numpy as np
import run as  r

'''
[id]
134

[name]
SGDRegressor

[input]
x_train	训练集	训练集标签数据集	二维数组	必须	定数
y_train	测试集	测试集数据集	二维数组	必须	定数
x_test	训练集标签	训练集标签标签	一维数组	必须	定数
y_test	测试集标签	测试集标签	一维数组	必须	定数
loss	损失函数	默认为'squared_loss',要使用的损失函数。squared_loss是指普通的最小二乘拟合。huber修改了squared_loss，通过将平方损失转换为线性损失超过ε距离，从而减少了对异常值校正的关注。epsilon_insensitive忽略小于epsilon的错误，并且超出此范围呈线性关系。squared_epsilon_insensitive是相同的，但超过ε容差后变为平方损耗,可选字符串	字符串	不必须	定数
penalty	惩罚规范	默认为'l2',要使用的惩罚规范(又称正则化术语)。默认值为l2，这是线性SVM模型的标准正则化器。l1和elasticnet可能会给模型带来稀疏性(特征选择)，而l2是无法实现的,可选l2,l1,elasticnet	字符串	不必须	定数
alpha	alpha	默认为0.0001,与正则项相乘的常数。值越高，正则化越强。当设置为learning_rate设置为optimal时，也用于计算学习率,可选浮点数	浮点数	不必须	定数
l1_ratio	l1_ratio	默认为0.15,0<=l1_ratio<=1的ElasticNet混合参数。l1_ratio=0对应于L2惩罚，l1_ratio=1到L1。仅在惩罚为elasticnet时使用,可选浮点数	浮点数	不必须	定数
fit_intercept	计算截距	默认为True,是否应该估计截距。如果为False，则假定数据已经居中,可选整数,布尔值	字符串	不必须	定数
max_iter	最大迭代次数	默认为1000,通过训练数据的最大次数(又称历元)。它只会影响fit方法的行为，而不会影响：meth：partial_fit方法的行为,可选整数	整数	不必须	定数
tol	tol	默认为1e-3,停止标准。如果不是None，则连续(n>_n_iter_no_change_epoch)的训练将在(loss>best_loss-tol)时停止,可选浮点数	浮点数	不必须	定数
shuffle	打乱	默认为True,在每个时期之后是否应重新整理训练数据,可选布尔值	布尔值	不必须	定数
verbose	详细程度	默认为0,详细程度,可选整数	整数	不必须	定数
epsilon	epsilon	默认为0.1,仅当损失为huber，epsilon_insensitive或squared_epsilon_insensitive时，为不敏感的损失函数。对于huber，确定使预测正确正确变得不那么重要的阈值。对于不敏感epsilon的情况，如果当前预测和正确标签之间的任何差异小于此阈值，则将忽略它们,可选浮点数	浮点数	不必须	定数
random_state	随机状态	默认为None,当shuffle设置为True时用于混洗数据。为多个函数调用传递可重复输出的int值,可选整数	整数	不必须	定数
learning_rate	学习率	默认为'invscaling',学习率时间表：-'constant'：eta=eta0；-optimal：eta=1.0/(alpha*(t+t0))其中t0由LeonBottou提出的启发式方法选择；-invscaling：eta=eta0/pow(t，power_t)；-adaptive：eta=eta0，只要训练持续减少即可。每次n_iter_no_change个连续的纪元未能将训练损失减少一倍，或者如果Early_stopping为True，则无法增加鉴定分数一倍，则当前学习率除以5,可选字符串	字符串	不必须	定数
eta0	初始学习率	默认为0.01,增量或自适应进度表的初始学习率，可选双精度浮点数	数字	不必须	定数
power_t	反比例学习率指数	默认为0.25,反比例学习率的指数,可选双精度浮点数	数字	不必须	定数
early_stopping	early_stopping	默认为False,当验证分数没有提高时是否使用提前停止来终止训练。如果设置为True，它将自动保留一小部分训练数据作为验证，并在连续nsit_no_change时期将score方法返回的验证分数提高至少tol时终止训练,可选布尔值	布尔值	不必须	定数
validation_fraction	验证分数	默认为0.1,预留的训练数据比例作为早期停止的验证集。必须在0到1之间。仅在early_stopping为True时使用,可选浮点数	浮点数	不必须	定数
n_iter_no_change	迭代不变次数	默认为5,迭代次数，没有改善，请提前停止,可选整数	整数	不必须	定数
warm_start	warm_start	默认为False,设置为True时，请重用上一个调用的解决方案以适合初始化，否则，只需擦除以前的解决方案即可。当warm_start为True时，重复调用fit或partial_fit可能会导致解决方案与一次调用fit时产生不同的解决方案。如果使用动态学习率，则根据已经看到的样本数量调整学习率,可选布尔值	布尔值	不必须	定数
average	平均值	默认为False,设置为True时，将计算所有更新的平均SGD权重并将结果存储在coef_属性中。如果将int设置为大于1，则一旦看到的样本总数达到平均值，就会开始平均。因此，平均=10将在看到10个样本后开始平均,可选整数,布尔值	字符串	不必须	定数

[output]
train_predict	预测	训练集预测结果	一维数组(数值)
test_predict	预测	测试集预测结果	一维数组(数值)
train_score	正确率	训练集预测结果的正确率	数字
test_score	正确率	测试集预测结果的正确率	数字
coef_	参数向量	分配给特征的权重	一维数组
intercept_	截距	拦截项	整数
n_iter_	迭代次数	达到停止标准之前的实际迭代次数	整数
t_	t_	训练期间进行的权重更新次数。与(n_iter_*n_samples)相同	整数

[outline]
通过使用SGD最小化正则经验损失来拟合线性模型

[describe]
通过使用SGD最小化正则经验损失来拟合线性模型
SGD代表随机梯度下降：损失的梯度为一次估算每个样本，并随即更新模型逐渐减少的力量计划(又称学习率)。
正则化器是对缩小模型的损失函数的惩罚使用平方欧几里德范数向零向量的参数L2或绝对范数L1或两者的组合(弹性网)。
如果由于正则化，参数更新越过0.0值，更新被截断为0.0以允许学习稀疏模型并实现在线特征选择。
此实现适用于表示为的密集numpy数组的数据特征的浮点值
'''


def main(x_train, y_train, x_test, y_test,
         loss="squared_loss", penalty="l2", alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=1e-3,
         shuffle=True, verbose=0, epsilon=0.1, random_state=None, learning_rate="invscaling", eta0=0.01,
         power_t=0.25, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, warm_start=False,
         average=False
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
                 random_state=random_state,
                 learning_rate=learning_rate,
                 eta0=eta0,
                 power_t=power_t,
                 early_stopping=early_stopping,
                 validation_fraction=validation_fraction,
                 n_iter_no_change=n_iter_no_change,
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
    back = main(x, y,x,y)

    print(back)
    for i in back:
        print(i + ":" + str(back[i]))

    json.dumps(back)
