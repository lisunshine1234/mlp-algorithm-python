import numpy as np
import run as  r

'''
[id]
99

[name]
Perceptron

[input]
x_train	训练集	训练集标签数据集	二维数组	必须	定数
y_train	测试集	测试集数据集	二维数组	必须	定数
x_test	训练集标签	训练集标签标签	一维数组	必须	定数
y_test	测试集标签	测试集标签	一维数组	必须	定数
penalty	penalty	默认为None,要使用的惩罚(又称正则化术语),可选'l2','l1','elasticnet'	字符串	不必须	定数
alpha	alpha	默认为0.0001,如果使用正则化，则乘以正则化项的常数,可选浮点数	浮点数	不必须	定数
fit_intercept	计算截距	默认为True,是否应该估计截距。如果为False，则假定数据已居中,可选整数,布尔值	字符串	不必须	定数
max_iter	最大迭代次数	默认为1000,通过训练数据的最大次数(又称历元)，它仅影响'fit'方法中的行为，而不影响：meth：'partial_fit'方法,可选整数	整数	不必须	定数
tol	tol	默认为1e-3,停止标准。如果不是None，则迭代将在(loss> previous_loss-tol)时停止,可选浮点数	浮点数	不必须	定数
shuffle	打乱	默认为True,在每个时期之后是否应重新整理训练数据,可选布尔值	布尔值	不必须	定数
verbose	详细程度	默认为0,详细程度,可选整数	整数	不必须	定数
eta0	初始学习率	默认为1.0,更新乘以的常数,可选双精度浮点数	双精度浮点数	不必须	定数
n_jobs	CPU数量	默认为None,用于执行OVA(一个对所有问题，对于多类问题)的CPU数量。除非在：obj：'None'上下文中，否则'joblib.parallel_backend'表示1。有关更多详细信息,可选整数	整数	不必须	定数
random_state	随机种子	默认为0,当'shuffle'设置为'True'时，用于混洗训练数据。在多个函数调用之间传递int以实现可重现的输出,可选整数	整数	不必须	定数
early_stopping	停止终止训练	默认为False,当validate.score没有改善时，是否使用提前停止来终止训练。如果设置为True，则当验证分数至少连续两个时期没有提高时，它将自动将训练数据的分层分数设置为验证并终止训练,可选布尔值	布尔值	不必须	定数
validation_fraction	验证分数	默认为0.1,预留的训练数据比例作为早早停止的验证集。必须介于0和1之间。仅当early_stopping为True时使用,可选浮点数	浮点数	不必须	定数
n_iter_no_change	迭代不变次数	默认为5,迭代次数，没有改善，请提前停止,可选整数	整数	不必须	定数
class_weight	类别权重	默认为None,预设为class_weight fit参数。与类关联的权重。如果未给出，则所有类都应具有权重1. 'balanced'模式使用y的值自动将权重与输入数据中的类频率成反比地调整为'n_samples / (n_classes * np.bincount(y)),可选'balanced'	字符串	不必须	定数
warm_start	warm_start	默认为False,设置为True时，请重新使用上一个调用的解决方案以适合初始化，否则，只需擦除以前的解决方案即可。请参阅：条款：'the Glossary <warm_start>',可选布尔值	布尔值	不必须	定数

[output]
train_predict	预测	训练集预测结果	一维数组(数值)
test_predict	预测	测试集预测结果	一维数组(数值)
train_score	正确率	训练集预测结果的正确率	数字
test_score	正确率	测试集预测结果的正确率	数字
coef_	参数向量	分配给特征的权重	二维数组
intercept_	截距	决策函数中的常量	一维数组
n_iter_	迭代次数	达到停止标准的实际迭代次数。对于多类拟合，它是每个二进制拟合中的最大值	整数
classes_	classes_	唯一的类标签	一维数组
t_	权重更新次数	训练期间进行的体重更新次数，与'(n_iter_ * n_samples)'相同	整数

[outline]
感知器

[describe]
感知器

'''


def main(x_train, y_train, x_test, y_test,
         penalty=None, alpha=0.0001, fit_intercept=True, max_iter=1000, tol=1e-3, shuffle=True, verbose=0, eta0=1.0, n_jobs=None, random_state=0,
         early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None, warm_start=False
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
    if type(eta0) is str:
        eta0 = eval(eta0)
    if type(n_jobs) is str:
        n_jobs = eval(n_jobs)
    if type(random_state) is str:
        random_state = eval(random_state)
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
    return r.run(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, penalty=penalty,
                 alpha=alpha,
                 fit_intercept=fit_intercept,
                 max_iter=max_iter,
                 tol=tol,
                 shuffle=shuffle,
                 verbose=verbose,
                 eta0=eta0,
                 n_jobs=n_jobs,
                 random_state=random_state,
                 early_stopping=early_stopping,
                 validation_fraction=validation_fraction,
                 n_iter_no_change=n_iter_no_change,
                 class_weight=class_weight,
                 warm_start=warm_start
                 )


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