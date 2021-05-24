import numpy as np
import run as  r

'''
[id]
114

[name]
ARDRegression

[input]
x_train	训练集	训练集标签数据集	二维数组	必须	定数
y_train	测试集	测试集数据集	二维数组	必须	定数
x_test	训练集标签	训练集标签标签	一维数组	必须	定数
y_test	测试集标签	测试集标签	一维数组	必须	定数
n_iter	n_iter	默认为300,最大迭代次数,可选整数	整数	不必须	定数
tol	tol	默认为1e-3,如果w收敛，则停止算法,可选浮点数	浮点数	不必须	定数
alpha_1	alpha_1	默认为1e-6,Hyper-parameter：shape参数，用于先于Alpha参数的Gamma分布,可选浮点数	浮点数	不必须	定数
alpha_2	alpha_2	默认为1e-6,超参数：Gamma分布优先于alpha参数的反比例参数(速率参数),可选浮点数	浮点数	不必须	定数
lambda_1	lambda_1	默认为1e-6,Hyper-parameter：shape参数，用于先于lambda参数的Gamma分布,可选浮点数	浮点数	不必须	定数
lambda_2	lambda_2	默认为1e-6,超参数：Gamma分布先于lambda参数的反比例参数(速率参数),可选浮点数	浮点数	不必须	定数
compute_score	compute_score	默认为False,如果为True，则在模型的每个步骤计算目标函数,可选布尔值	布尔值	不必须	定数
threshold_lambda	threshold_lambda	默认为10000,从计算中高精度删除(修剪)权重的阈值,可选浮点数	浮点数	不必须	定数
fit_intercept	计算截距	默认为True,是否计算该模型的截距。如果设置为false，则在计算中将不使用截距(即，数据应居中),可选整数,布尔值	字符串	不必须	定数
normalize	归一化	默认为False,当fit_intercept设置为False时，将忽略此参数。如果为True，则将在回归之前通过减去均值并除以l2-范数来对回归变量X进行归一化,可选布尔值	布尔值	不必须	定数
copy_X	是否复制	默认为True,如果为True，将复制X；否则为X。否则，它可能会被覆盖,可选布尔值	布尔值	不必须	定数
verbose	详细程度	默认为False,拟合模型时为详细模式,可选布尔值	布尔值	不必须	定数

[output]
train_predict	预测	训练集预测结果	一维数组(数值)
test_predict	预测	测试集预测结果	一维数组(数值)
train_score	正确率	训练集预测结果的正确率	数字
test_score	正确率	测试集预测结果的正确率	数字
coef_	参数向量	回归模型的系数(均值)	一维数组
alpha_	alpha	估计的噪声精度	浮点数
lambda_	lambda_	重量的估计精度	一维数组
sigma_	sigma_	权重的估计方差-协方差矩阵	二维数组
scores_	scores_	如果计算，则目标函数的值(将最大化)	浮点数
intercept_	截距	决策特征中的独立术语。如果fit_intercept=False，则设置为0.0	整数

[outline]
贝叶斯ARD回归。

[describe]
贝叶斯ARD回归。
使用ARD事前拟合回归模型的权重。
的重量回归模型假定为高斯分布。
还要估计参数lambda(权重)和alpha(噪声分布的精度)。
通过迭代过程(证据最大化)进行估算
'''


def main(x_train, y_train, x_test, y_test,
         n_iter=300, tol=1.e-3, alpha_1=1.e-6, alpha_2=1.e-6, lambda_1=1.e-6, lambda_2=1.e-6, compute_score=False,
         threshold_lambda=1.e+4, fit_intercept=True, normalize=False, copy_X=True, verbose=False
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
    if type(compute_score) is str:
        compute_score = eval(compute_score)
    if type(threshold_lambda) is str:
        threshold_lambda = eval(threshold_lambda)
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
                 compute_score=compute_score,
                 threshold_lambda=threshold_lambda,
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
