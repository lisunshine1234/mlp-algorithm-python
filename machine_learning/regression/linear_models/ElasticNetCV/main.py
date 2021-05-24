import numpy as np
import run as  r

'''
[id]
117

[name]
ElasticNetCV

[input]
x_train	训练集	训练集标签数据集	二维数组	必须	定数
y_train	测试集	测试集数据集	二维数组	必须	定数
x_test	训练集标签	训练集标签标签	一维数组	必须	定数
y_test	测试集标签	测试集标签	一维数组	必须	定数
l1_ratio	l1_ratio	默认为0.5,在0和1之间的float传递给ElasticNet(在l1和l2罚分之间缩放)。当l1_ratio=0时，惩罚为L2惩罚。对于l1_ratio=1，这是L1惩罚。对于0<l1_ratio<1，惩罚是L1和L2的组合。此参数可以是列表，在这种情况下，通过交叉验证测试不同的值，并使用给出最佳预测得分的值,可选浮点数	浮点数	不必须	定数
eps	eps	默认为1e-3,路径的长度。eps=1e-3表示alpha_min/alpha_max=1e-3,可选浮点数	浮点数	不必须	定数
n_alphas	alpha数	默认为100,沿正则化路径的Alpha数，用于每个l1_ratio,可选整数	整数	不必须	定数
alphas	alpha列表	默认为None,用于计算模型的Alpha列表。如果没有自动设置Alpha,可选数组	一维数组	不必须	定数
fit_intercept	计算截距	默认为True,是否计算该模型的截距。如果设置为false，则在计算中将不使用截距(即，数据应居中),可选整数,布尔值	字符串	不必须	定数
normalize	归一化	默认为False,当fit_intercept设置为False时，将忽略此参数。如果为True，则将在回归之前通过减去均值并除以l2-范数来对回归变量X进行归一化,可选布尔值	布尔值	不必须	定数
precompute	预先计算	默认为auto,default=auto是否使用预先计算的Gram矩阵来加快计算速度。如果设置为auto让我们决定。语法矩阵也可以作为参数传递,可选布尔值,数组	字符串	不必须	定数
max_iter	最大迭代次数	默认为1000,最大迭代次数,可选整数	整数	不必须	定数
tol	tol	默认为1e-4,优化的容忍度：如果更新小于tol，则优化代码将检查双重间隙的最佳性，并继续进行直到其小于tol,可选浮点数	浮点数	不必须	定数
cv	交叉验证	默认为None,确定交叉验证拆分策略。cv的可能输入是：-无，使用默认的5倍交叉验证，-int指定折叠数。-CVsplitter，-可迭代的产量(训练，测试)拆分为索引数组。对于int/None输入，使用：KFold类。有关各种信息，请参考：用户指南<cross_validation>,可选整数	整数	不必须	定数
copy_X	是否复制	默认为True,如果为True，将复制X;否则，它可能会被覆盖,可选布尔值	布尔值	不必须	定数
verbose	详细程度	默认为0,详细程度,可选整数,布尔值	字符串	不必须	定数
n_jobs	CPU数量	默认为None,交叉验证期间要使用的CPU数量。除非在：obj：joblib.parallel_backend上下文中，否则None表示1。更多细节,可选整数	整数	不必须	定数
positive	positive	默认为False,设置为True时，强制系数为正,可选布尔值	布尔值	不必须	定数
random_state	随机状态	默认为None,选择随机特征进行更新的伪随机数生成器的种子。在selection==random时使用。为多个函数调用传递可重复输出的int值,可选整数	整数	不必须	定数
selection	selection	默认为'cyclic',如果设置为随机，则随机系数将在每次迭代时更新，而不是默认情况下按顺序遍历特征。这(设置为随机)通常会导致收敛更快，尤其是当tol高于1e-4时,可选cyclic,random	字符串	不必须	定数

[output]
train_predict	预测	训练集预测结果	一维数组(数值)
test_predict	预测	测试集预测结果	一维数组(数值)
train_score	正确率	训练集预测结果的正确率	数字
test_score	正确率	测试集预测结果的正确率	数字
alpha_	alpha	通过交叉验证选择的惩罚量	浮点数
l1_ratio_	l1_ratio_	通过交叉验证在l1和l2惩罚之间折衷选择	浮点数
coef_	参数向量	参数向量(成本函数公式中的w)	一维数组
intercept_	截距	决策函数中的独立项	整数
mse_path_	均方误差	每个折叠上测试集的均方误差，变化的l1_ratio和alpha	二维数组
alphas_	alpha网格	对于每个l1_ratio，用于拟合的alpha网格	一维数组
n_iter_	迭代次数	坐标下降求解器运行的迭代次数，以达到最佳alpha的指定公差	整数

[outline]
沿着正则化路径具有迭代拟合的ElasticNet模型。

[describe]
沿着正则化路径具有迭代拟合的ElasticNet模型。
'''


def main(x_train, y_train, x_test, y_test,
         l1_ratio=0.5, eps=1e-3, n_alphas=100, alphas=None, fit_intercept=True, normalize=False, precompute='auto',
         max_iter=1000, tol=1e-4, cv=None, copy_X=True, verbose=0, n_jobs=None, positive=False, random_state=None,
         selection='cyclic'
         ):
    if type(l1_ratio) is str:
        l1_ratio = eval(l1_ratio)
    if type(eps) is str:
        eps = eval(eps)
    if type(n_alphas) is str:
        n_alphas = eval(n_alphas)
    if type(alphas) is str:
        alphas = eval(alphas)
    if type(fit_intercept) is str:
        fit_intercept = eval(fit_intercept)
    if type(normalize) is str:
        normalize = eval(normalize)
    if type(precompute) is str and precompute != 'auto':
        precompute = eval(precompute)
    if type(max_iter) is str:
        max_iter = eval(max_iter)
    if type(tol) is str:
        tol = eval(tol)
    if type(cv) is str:
        cv = eval(cv)
    if type(copy_X) is str:
        copy_X = eval(copy_X)
    if type(verbose) is str:
        verbose = eval(verbose)
    if type(n_jobs) is str:
        n_jobs = eval(n_jobs)
    if type(positive) is str:
        positive = eval(positive)
    if type(random_state) is str:
        random_state = eval(random_state)

    return r.run(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                 l1_ratio=l1_ratio,
                 eps=eps,
                 n_alphas=n_alphas,
                 alphas=alphas,
                 fit_intercept=fit_intercept,
                 normalize=normalize,
                 precompute=precompute,
                 max_iter=max_iter,
                 tol=tol,
                 cv=cv,
                 copy_X=copy_X,
                 verbose=verbose,
                 n_jobs=n_jobs,
                 positive=positive,
                 random_state=random_state,
                 selection=selection)


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
