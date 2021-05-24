import numpy as np
import run as  r

'''
[id]
115

[name]
BayesianRidge

[input]
x_train	训练集	训练集标签数据集	二维数组	必须	定数
y_train	测试集	测试集数据集	二维数组	必须	定数
x_test	训练集标签	训练集标签标签	一维数组	必须	定数
y_test	测试集标签	测试集标签	一维数组	必须	定数
n_iter	n_iter	默认为300,最大迭代次数。应该大于或等于1,可选整数	整数	不必须	定数
tol	tol	默认为1e-3,如果w收敛，则停止算法,可选浮点数	浮点数	不必须	定数
alpha_1	alpha_1	默认为1e-6,Hyper-parameter：shape参数，用于先于Alpha参数的Gamma分布,可选浮点数	浮点数	不必须	定数
alpha_2	alpha_2	默认为1e-6,超参数：Gamma分布优先于alpha参数的反比例参数(速率参数),可选浮点数	浮点数	不必须	定数
lambda_1	lambda_1	默认为1e-6,Hyper-parameter：shape参数，用于先于lambda参数的Gamma分布,可选浮点数	浮点数	不必须	定数
lambda_2	lambda_2	默认为1e-6,超参数：Gamma分布先于lambda参数的反比例参数(速率参数),可选浮点数	浮点数	不必须	定数
alpha_init	alpha_init	默认为None,alpha的初始值(噪声的精度)。如果未设置，则alpha_init为1/Var(y),可选浮点数	浮点数	不必须	定数
lambda_init	lambda_init	默认为None,Lambda的初始值(权重的精度)。如果未设置，则lambda_init为1。..版本添加：：0.22,可选浮点数	浮点数	不必须	定数
compute_score	compute_score	默认为False,如果为True，则在每次优化迭代时计算对数边际可能性,可选布尔值	布尔值	不必须	定数
fit_intercept	计算截距	默认为True,是否计算此模型的截距。截距不被视为概率参数，因此没有关联的方差。如果将其设置为False，则在计算中将不使用截距(即，数据应居中),可选整数,布尔值	字符串	不必须	定数
normalize	归一化	默认为False,当fit_intercept设置为False时，将忽略此参数。如果为True，则将在回归之前通过减去均值并除以l2-范数来对回归变量X进行归一化,可选布尔值	布尔值	不必须	定数
copy_X	是否复制	默认为True,如果为True，将复制X；否则为X。否则，它可能会被覆盖,可选布尔值	布尔值	不必须	定数
verbose	详细程度	默认为False,拟合模型时为详细模式,可选布尔值	布尔值	不必须	定数

[output]
train_predict	预测	训练集预测结果	一维数组(数值)
test_predict	预测	测试集预测结果	一维数组(数值)
train_score	正确率	训练集预测结果的正确率	数字
test_score	正确率	测试集预测结果的正确率	数字
coef_	参数向量	回归模型的系数(均值)	一维数组
intercept_	截距	决策特征中的独立术语。如果fit_intercept=False，则设置为0.0	整数
alpha_	alpha	估计的噪声精度	浮点数
lambda_	lambda_	估计重量的精度	浮点数
sigma_	sigma_	权重的估计方差-协方差矩阵	二维数组
scores_	scores_	如果calculated_score为True，则在每次优化迭代时对数边际似然值(要最大化)。该数组以从alpha和lambda的初始值获得的对数边际似然值开始，以以估计的alpha和lambda的值结束	一维数组
n_iter_	迭代次数	达到停止标准的实际迭代次数	整数

[outline]
贝叶斯岭回归。

[describe]
贝叶斯岭回归。
拟合贝叶斯岭模型。
有关详细信息，请参见注释部分。
正则化参数的实现和优化lambda(权重的精度)和alpha(噪声的精度)
'''


def main(x_train, y_train, x_test, y_test,
         n_iter=300, tol=1.e-3, alpha_1=1.e-6, alpha_2=1.e-6, lambda_1=1.e-6, lambda_2=1.e-6, alpha_init=None,
         lambda_init=None, compute_score=False, fit_intercept=True, normalize=False, copy_X=True, verbose=False
         ):
    if type(x_train) is str:
        x_train = eval(x_train)
    if type(y_train) is str:
        y_train = eval(y_train)
    if type(x_test) is str:
        x_test = eval(x_test)
    if type(y_test) is str:
        y_test = eval(y_test)
    if type(n_iter) is str:
        n_iter = eval(n_iter)
    if type(tol) is str:
        tol = eval(tol)
    if type(alpha_1) is str:
        alpha_1 = eval(alpha_1)
    if type(alpha_2) is str:
        alpha_2 = eval(alpha_2)
    if type(lambda_1) is str:
        lambda_1 = eval(lambda_1)
    if type(lambda_2) is str:
        lambda_2 = eval(lambda_2)
    if type(alpha_init) is str:
        alpha_init = eval(alpha_init)
    if type(lambda_init) is str:
        lambda_init = eval(lambda_init)
    if type(compute_score) is str:
        compute_score = eval(compute_score)
    if type(fit_intercept) is str:
        fit_intercept = eval(fit_intercept)
    if type(normalize) is str:
        normalize = eval(normalize)
    if type(copy_X) is str:
        copy_X = eval(copy_X)
    if type(verbose) is str:
        verbose = eval(verbose)

    return r.run(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, n_iter=n_iter,
                 tol=tol,
                 alpha_1=alpha_1,
                 alpha_2=alpha_2,
                 lambda_1=lambda_1,
                 lambda_2=lambda_2,
                 alpha_init=alpha_init,
                 lambda_init=lambda_init,
                 compute_score=compute_score,
                 fit_intercept=fit_intercept,
                 normalize=normalize,
                 copy_X=copy_X,
                 verbose=verbose)


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

