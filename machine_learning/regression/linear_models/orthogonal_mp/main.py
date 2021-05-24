import numpy                as                np
import run                as                r

'''
[id]
128

[name]
orthogonal_mp

[input]
x_train	训练集	训练集标签数据集	二维数组	必须	定数
y_train	测试集	测试集数据集	二维数组	必须	定数
x_test	训练集标签	训练集标签标签	一维数组	必须	定数
y_test	测试集标签	测试集标签	一维数组	必须	定数
n_nonzero_coefs	非零条目数	默认为None,解决方案中所需的非零条目数。如果为None(默认)，则此值设置为n_features的10％,可选整数	整数	不必须	定数
tol	tol	默认为None,残数的最大范数。如果不是None，则覆盖n_nonzero_coefs,可选浮点数	浮点数	不必须	定数
precompute	预先计算	默认为False,是否执行预计算。当n_targets或n_samples非常大时提高性能,可选布尔值,字符串	字符串	不必须	定数
copy_X	是否复制	默认为True,设计矩阵X是否必须由算法复制。仅当X已被Fortran排序时，false值才有用，否则无论如何都会进行复制,可选布尔值	布尔值	不必须	定数
return_path	return_path	默认为False,是否沿前向路径返回非零系数的每个值。对于交叉验证很有用,可选布尔值	布尔值	不必须	定数
return_n_iter	return_n_iter	默认为False,是否返回迭代次数,可选布尔值	布尔值	不必须	定数

[output]
train_predict	预测	训练集预测结果	一维数组(数值)
test_predict	预测	测试集预测结果	一维数组(数值)
train_score	正确率	训练集预测结果的正确率	数字
test_score	正确率	测试集预测结果的正确率	数字
coef	coef	OMP解决方案的系数。如果return_path=True，则包含整个系数路径。在这种情况下，其形状为(n_features，n_features)或(n_features，n_targets，n_features)，并且在最后一个轴上进行迭代会按活动特征的递增顺序生成系数	一维数组
n_iters	n_iters	每个目标上的活动特征数量。仅在return_n_iter设置为True时返回	整数

[outline]
正交匹配追踪(OMP)解决n_targets正交匹配追踪问题。

[describe]
正交匹配追踪(OMP)解决n_targets正交匹配追踪问题。
问题的实例具有以下形式：当使用非零系数的数量参数化时，使用n_nonzero_coefs：argmin||y-X\gamma||^2服从||\gamma||_0<=n_{nonzerocoefs}当使用参数tol进行参数设置时：argmin||\gamma||_0服从||y-X\gamma||^2<=tol
'''


def main(x_train, y_train,
         n_nonzero_coefs=None, tol=None, precompute=False, copy_X=True, return_path=False, return_n_iter=False):
    if type(x_train) is str:
        x_train = eval(x_train)
    if type(y_train) is str:
        y_train = eval(y_train)
    if type(n_nonzero_coefs) is str:
        n_nonzero_coefs = eval(n_nonzero_coefs)
    if type(tol) is str:
        tol = eval(tol)
    if type(precompute) is str and precompute != 'auto':
        precompute = eval(precompute)
    if type(copy_X) is str:
        copy_X = eval(copy_X)
    if type(return_path) is str:
        return_path = eval(return_path)
    if type(return_n_iter) is str:
        return_n_iter = eval(return_n_iter)
    return r.run(x_train=x_train, y_train=y_train,  n_nonzero_coefs=n_nonzero_coefs,
                 tol=tol,
                 precompute=precompute,
                 copy_X=copy_X,
                 return_path=return_path,
                 return_n_iter=return_n_iter)



if __name__ == '__main__':
    import numpy as np
    import json

    array = np.loadtxt('D:\\123_2.csv', delimiter=',')
    array = array[0:20, :]

    y = array[:, -1].tolist()
    x = np.delete(array, -1, axis=1).tolist()
    array = array.tolist()
    back = main(x, y)

    print(back)
    for i in back:
        print(i + ":" + str(back[i]))

    json.dumps(back)
