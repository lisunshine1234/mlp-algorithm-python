import numpy    as    np
import run    as    r

'''
[id]
127

[name]
LinearRegression

[input]
x_train	训练集	训练集标签数据集	二维数组	必须	定数
y_train	测试集	测试集数据集	二维数组	必须	定数
x_test	训练集标签	训练集标签标签	一维数组	必须	定数
y_test	测试集标签	测试集标签	一维数组	必须	定数
fit_intercept	截距	默认为True,是否计算此模型的截距。如果将其设置为False，则在计算中将不使用截距(即，数据应居中),可选整数,布尔值	字符串	不必须	定数
normalize	归一化	默认为False,当fit_intercept设置为False时，将忽略此参数。如果为True，则将在回归之前通过减去均值并除以l2-范数来对回归变量X进行归一化,可选布尔值	布尔值	不必须	定数
copy_X	是否复制	默认为True,如果为True，将复制X；否则为X。否则，它可能会被覆盖,可选布尔值	布尔值	不必须	定数
n_jobs	作业数	默认为None,用于计算的作业数。这只会为n_targets>1和足够大的问题提供加速。除非在：obj：joblib.parallel_backend上下文中，否则None表示1。-1表示使用所有处理器,可选整数	整数	不必须	定数

[output]
train_predict	预测	训练集预测结果	一维数组(数值)
test_predict	预测	测试集预测结果	一维数组(数值)
train_score	正确率	训练集预测结果的正确率	数字
test_score	正确率	测试集预测结果的正确率	数字
coef_	估计系数	线性回归问题的估计系数。如果在拟合过程中传递了多个目标(y2D)，则这是一个2D形状的数组(n_targets，n_features)，而如果仅传递了一个目标，则这是长度为n_features的一维数组	二维数组
rank_	等级	矩阵的等级x。仅在x密集时可用	整数
singular_	奇异值	奇异值x。仅在x密集时可用	一维数组
intercept_	截距	线性模型中的独立项。如果设置为0.0。fit_intercept=False	整数

[outline]
普通最小二乘线性回归。

[describe]
普通最小二乘线性回归。
	LinearRegression使用系数w	=（w1，...，wp）拟合线性模型，以最小化数据集中观察到的目标与通过线性近似预测的目标之间的平方余数。
'''


def main(x_train, y_train, x_test, y_test, fit_intercept=True, normalize=False, copy_X=True, n_jobs=None):
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
    if type(normalize) is str:
        normalize = eval(normalize)
    if type(copy_X) is str:
        copy_X = eval(copy_X)
    if type(n_jobs) is str:
        n_jobs = eval(n_jobs)

    return r.run(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, fit_intercept=fit_intercept,
                 normalize=normalize,
                 copy_X=copy_X,
                 n_jobs=n_jobs)



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
