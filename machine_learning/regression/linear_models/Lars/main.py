import numpy as np
import run as  r

'''
[id]
123

[name]
Lars

[input]
x_train	训练集	训练集标签数据集	二维数组	必须	定数
y_train	测试集	测试集数据集	二维数组	必须	定数
x_test	训练集标签	训练集标签标签	一维数组	必须	定数
y_test	测试集标签	测试集标签	一维数组	必须	定数
fit_intercept	计算截距	默认为True,是否计算此模型的截距。如果设置为false，则在计算中将不使用截距(即，数据应居中),可选整数,布尔值	字符串	不必须	定数
verbose	详细程度	默认为False,设置详细程度,可选整数,布尔值	字符串	不必须	定数
normalize	归一化	默认为True,当fit_intercept设置为False时，将忽略此参数。如果为True，则将在回归之前通过减去均值并除以l2-范数来对回归变量X进行归一化,可选布尔值	布尔值	不必须	定数
precompute	预先计算	默认为'auto',是否使用预先计算的Gram矩阵来加快计算速度。如果设置为auto让我们决定。语法矩阵也可以作为参数传递,可选布尔值,数组	字符串	不必须	定数
n_nonzero_coefs	非零系数数量	默认为500,非零系数的目标数量。无限使用np.inf,可选整数	整数	不必须	定数
eps	eps	默认为np.finfo(np.float).eps,Cholesky对角线因子计算中的机器精度正则化。对于条件非常恶劣的系统，请增加此值。与某些基于迭代优化的算法中的tol参数不同，此参数不控制优化的容差。默认情况下使用np.finfo(np.float).eps,可选浮点数	浮点数	不必须	定数
copy_X	是否复制	默认为True,如果为True，将复制X;否则，它可能会被覆盖,可选布尔值	布尔值	不必须	定数
fit_path	fit路径	默认为True,如果为True，则完整路径存储在coef_path_属性中。如果您针对一个大问题或多个目标计算解决方案，则将fit_path设置为False将导致加速，尤其是在使用较小的alpha值时,可选布尔值	布尔值	不必须	定数
jitter	噪声参数上限	默认为None,要添加到y值的统一噪声参数的上限，以满足模型一次计算的假设。可能会对稳定性有所帮助,可选浮点数	浮点数	不必须	定数
random_state	随机状态	默认为,确定用于抖动的随机数生成。为多个函数调用传递可重复输出的int值,可选整数	整数	不必须	定数

[output]
train_predict	预测	训练集预测结果	一维数组(数值)
test_predict	预测	测试集预测结果	一维数组(数值)
train_score	正确率	训练集预测结果的正确率	数字
test_score	正确率	测试集预测结果的正确率	数字
alphas_	alpha网格	数组每次迭代的最大协方差(绝对值)。_alphas可以是n_nonzero_coefs或n_features，以较小者为准	一维数组
active_	active_	路径末尾的活动变量的索引	一维数组
coef_path_	coef_path_	|n_targets这样的数组的列表沿路径的系数的变化值。如果fit_path参数为False，则不存在	二维数组
coef_	参数向量	参数向量(配方公式中的w)	一维数组
intercept_	截距	决策特征中的独立术语	整数
n_iter_	迭代次数	lars_path为每个目标查找alpha网格所花费的迭代次数	整数

[outline]
最小角度回归模型LAR

[describe]
最小角度回归模型LAR
'''


def main(x_train, y_train, x_test, y_test,
         fit_intercept=True, verbose=False, normalize=True, precompute='auto', n_nonzero_coefs=500,
         eps=np.finfo(np.float).eps, copy_X=True, fit_path=True, jitter=None, random_state=None
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
    if type(normalize) is str:
        normalize = eval(normalize)
    if type(precompute) is str and precompute != 'auto':
        precompute = eval(precompute)
    if type(n_nonzero_coefs) is str:
        n_nonzero_coefs = eval(n_nonzero_coefs)
    if type(eps) is str:
        eps = eval(eps)
    if type(copy_X) is str:
        copy_X = eval(copy_X)
    if type(fit_path) is str:
        fit_path = eval(fit_path)
    if type(jitter) is str:
        jitter = eval(jitter)
    if type(random_state) is str:
        random_state = eval(random_state)

    return r.run(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, fit_intercept=fit_intercept,
                 verbose=verbose,
                 normalize=normalize,
                 precompute=precompute,
                 n_nonzero_coefs=n_nonzero_coefs,
                 eps=eps,
                 copy_X=copy_X,
                 fit_path=fit_path,
                 jitter=jitter,
                 random_state=random_state)

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
