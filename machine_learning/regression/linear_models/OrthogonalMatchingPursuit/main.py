import numpy as np
import run as  r

'''
[id]
129

[name]
OrthogonalMatchingPursuit

[input]
x_train	训练集	训练集标签数据集	二维数组	必须	定数
y_train	测试集	测试集数据集	二维数组	必须	定数
x_test	训练集标签	训练集标签标签	一维数组	必须	定数
y_test	测试集标签	测试集标签	一维数组	必须	定数
n_nonzero_coefs	非零条目数	默认为None,解决方案中所需的非零条目数。如果为None(默认)，则此值设置为n_features的10％,可选整数	整数	不必须	定数
tol	tol	默认为None,残数的最大范数。如果不是None，则覆盖n_nonzero_coefs,可选浮点数	浮点数	不必须	定数
fit_intercept	计算截距	默认为True,是否计算该模型的截距。如果设置为false，则在计算中将不使用截距(即，数据应居中),可选整数,布尔值	字符串	不必须	定数
normalize	归一化	默认为True,当fit_intercept设置为False时，将忽略此参数。如果为True，则将在回归之前通过减去均值并除以l2-范数来对回归变量X进行归一化,可选布尔值	布尔值	不必须	定数
precompute	预先计算	默认为'auto',是否使用预先计算的Gram和Xy矩阵来加快计算速度。当n_targets或n_samples非常大时提高性能。请注意，如果您已经有这样的矩阵，则可以将它们直接传递给fit方法,可选布尔值，字符串	字符串	不必须	定数

[output]
train_predict	预测	训练集预测结果	一维数组(数值)
test_predict	预测	测试集预测结果	一维数组(数值)
train_score	正确率	训练集预测结果的正确率	数字
test_score	正确率	测试集预测结果的正确率	数字
coef_	参数向量	参数向量(公式中的w)	一维数组
intercept_	截距	决策函数中的独立项	整数
n_iter_	迭代次数	每个目标上的活动特征数量	整数

[outline]
正交匹配追踪模型(OMP)

[describe]
正交匹配追踪模型(OMP)
'''


def main(x_train, y_train, x_test, y_test,
         n_nonzero_coefs=None, tol=None, fit_intercept=True, normalize=True, precompute='auto'
         ):
    if type(x_train) is str:
        x_train = eval(x_train)
    if type(y_train) is str:
        y_train = eval(y_train)
    if type(x_test) is str:
        x_test = eval(x_test)
    if type(y_test) is str:
        y_test = eval(y_test)
    if type(n_nonzero_coefs) is str:
        n_nonzero_coefs = eval(n_nonzero_coefs)
    if type(tol) is str:
        tol = eval(tol)
    if type(fit_intercept) is str:
        fit_intercept = eval(fit_intercept)
    if type(normalize) is str:
        normalize = eval(normalize)
    if type(precompute) is str and precompute != 'auto':
        precompute = eval(precompute)
    return r.run(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                 n_nonzero_coefs=n_nonzero_coefs,
                 tol=tol,
                 fit_intercept=fit_intercept,
                 normalize=normalize,
                 precompute=precompute)


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
