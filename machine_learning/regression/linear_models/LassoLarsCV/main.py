import numpy as np
import run as  r

'''
[id]
122

[name]
LassoLarsCV

[input]
x_train	训练集	训练集标签数据集	二维数组	必须	定数
y_train	测试集	测试集数据集	二维数组	必须	定数
x_test	训练集标签	训练集标签标签	一维数组	必须	定数
y_test	测试集标签	测试集标签	一维数组	必须	定数
fit_intercept	计算截距	默认为True,是否计算该模型的截距。如果设置为false，则在计算中将不使用截距(即，数据应居中),可选整数,布尔值	字符串	不必须	定数
verbose	详细程度	默认为False,设置详细程度,可选整数,布尔值	字符串	不必须	定数
max_iter	最大迭代次数	默认为500,要执行的最大迭代次数,可选整数	整数	不必须	定数
normalize	归一化	默认为True,当fit_intercept设置为False时，将忽略此参数。如果为True，则将在回归之前通过减去均值并除以l2-范数来对回归变量X进行归一化。如果您希望标准化，请在使用normalize=False的估算器上调用fit之前,可选布尔值	布尔值	不必须	定数
precompute	预先计算	默认为'auto',是否使用预先计算的Gram矩阵来加快计算速度。如果设置为auto让我们决定。不能将Gram矩阵作为参数传递，因为我们将仅使用X的子集,可选布尔值	布尔值	不必须	定数
cv	交叉验证	默认为None,确定交叉验证拆分策略。cv的可能输入是：-无，使用默认的5倍交叉验证，-整数，指定折叠数。-CVsplitter，-可迭代的产量(训练，测试)拆分为索引数组。对于整数/无输入，使用：KFold类,可选整数	整数	不必须	定数
max_n_alphas	最大残差路径	默认为1000,交叉验证中用于计算残差的路径上的最大点数,可选整数	整数	不必须	定数
n_jobs	CPU数量	默认为None,交叉验证期间要使用的CPU数量。除非在：obj：joblib.parallel_backend上下文中，否则None表示1,可选整数	整数	不必须	定数
eps	机器精度正则化	默认对角线因子计算中的机器精度正则化。对于条件非常恶劣的系统，请增加此值,可选浮点数	浮点数	不必须	定数
copy_X	是否复制	默认为True,如果为True，将复制X；否则为X。否则，它可能会被覆盖,可选布尔值	布尔值	不必须	定数
positive	positive	默认为False,将系数限制为>=0。请注意，您可能希望删除默认设置为True的fit_intercept。在正约束下，对于较小的alpha值，模型系数不会收敛到普通最小二乘解。通常，逐步Lars-Lasso算法所达到的系数不超过最小alpha值(alphas_[alphas_>0。]。min())通常与坐标下降Lasso估计器的解一致。因此，使用LassoLarsCV仅对预期和/或达到稀疏解决方案的问题有意义,可选布尔值	布尔值	不必须	定数

[output]
train_predict	预测	训练集预测结果	一维数组(数值)
test_predict	预测	测试集预测结果	一维数组(数值)
train_score	正确率	训练集预测结果的正确率	数字
test_score	正确率	测试集预测结果的正确率	数字
coef_	参数向量	参数向量(公式中的w)	一维数组
intercept_	截距	决策函数中的独立项	整数
coef_path_	coef_path_	沿路径的系数的变化值	二维数组
alpha_	alpha	估计的正则化参数alpha	浮点数
alphas_	alpha网格	沿路径的不同alpha值	一维数组
cv_alphas_	cv_alphas_	沿路径的所有折叠的所有Alpha值	一维数组
mse_path_	均方误差	沿路径的每个折叠在左侧均方误差(cv_alphas给出的alpha值)	二维数组
n_iter_	迭代次数	Lars使用最佳alpha进行的迭代次数	整数

[outline]
使用LARS算法进行交叉验证的套索。

[describe]
使用LARS算法进行交叉验证的套索。
请参阅术语表：交叉验证估计器。
套索的优化目标是：(1/(2*n_samples))*||y-Xw||^2_2+alpha*||w||_1
'''


def main(x_train, y_train, x_test, y_test,
         fit_intercept=True, verbose=False, max_iter=500, normalize=True, precompute='auto', cv=None, max_n_alphas=1000,
         n_jobs=None, eps=np.finfo(np.float).eps, copy_X=True, positive=False
         ):
    if type(x_train) is str:
        x_train = eval(x_train)
    if type(y_train) is str:
        y_train = eval(y_train)
    if type(x_test) is str:
        x_test = eval(x_test)
    if type(y_test) is str:
        y_test = eval(y_test)
    if type(fit_intercept) is str:
        fit_intercept = eval(fit_intercept)
    if type(verbose) is str:
        verbose = eval(verbose)
    if type(max_iter) is str:
        max_iter = eval(max_iter)
    if type(normalize) is str:
        normalize = eval(normalize)
    if type(precompute) is str and precompute != 'auto':
        precompute = eval(precompute)
    if type(cv) is str:
        cv = eval(cv)
    if type(max_n_alphas) is str:
        max_n_alphas = eval(max_n_alphas)
    if type(n_jobs) is str:
        n_jobs = eval(n_jobs)
    if type(eps) is str:
        eps = eval(eps)
    if type(copy_X) is str:
        copy_X = eval(copy_X)
    if type(positive) is str:
        positive = eval(positive)

    return r.run(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                 fit_intercept=fit_intercept,
                 verbose=verbose,
                 max_iter=max_iter,
                 normalize=normalize,
                 precompute=precompute,
                 cv=cv,
                 max_n_alphas=max_n_alphas,
                 n_jobs=n_jobs,
                 eps=eps,
                 copy_X=copy_X,
                 positive=positive)



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
