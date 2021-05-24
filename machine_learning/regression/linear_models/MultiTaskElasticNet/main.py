import numpy as np
import run as  r

'''
[id]
125

[name]
MultiTaskElasticNet

[input]
x_train	训练集	训练集标签数据集	二维数组	必须	定数
y_train	测试集	测试集数据集	二维数组	必须	定数
x_test	训练集标签	训练集标签标签	一维数组	必须	定数
y_test	测试集标签	测试集标签	一维数组	必须	定数
alpha	alpha	默认为1.0,与L1/L2项相乘的常数。默认为1.0,可选浮点数	浮点数	不必须	定数
l1_ratio	l1_ratio	默认为0.5,ElasticNet混合参数，其中0<l1_ratio<=1。对于l1_ratio=1，惩罚是L1/L2惩罚。对于l1_ratio=0，这是L2损失。对于0<l1_ratio<1，惩罚是L1/L2和L2的组合,可选浮点数	浮点数	不必须	定数
fit_intercept	计算截距	默认为True,是否计算该模型的截距。如果设置为false，则在计算中将不使用截距(即，数据应居中),可选整数,布尔值	字符串	不必须	定数
normalize	归一化	默认为False,当fit_intercept设置为False时，将忽略此参数。如果为True，则将在回归之前通过减去均值并除以l2-范数来对回归变量X进行归一化,可选布尔值	布尔值	不必须	定数
copy_X	是否复制	默认为True,如果为True，将复制X;否则，它可能会被覆盖,可选布尔值	布尔值	不必须	定数
max_iter	最大迭代次数	默认为1000,最大迭代次数,可选整数	整数	不必须	定数
tol	tol	默认为1e-4,优化的容忍度：如果更新小于tol，则优化代码将检查双重间隙的最佳性，并继续进行直到其小于tol,可选浮点数	浮点数	不必须	定数
warm_start	warm_start	默认为False,当设置为True时，请重用上一个调用的解决方案以适合初始化，否则，只需擦除先前的解决方案即可,可选布尔值	布尔值	不必须	定数
random_state	随机状态	默认为None,选择随机特征进行更新的伪随机数生成器的种子。在selection==random时使用。为多个函数调用传递可重复输出的int值,可选整数	整数	不必须	定数
selection	selection	默认为'cyclic',如果设置为随机，则随机系数将在每次迭代时更新，而不是默认情况下按顺序遍历特征。这(设置为随机)通常会导致收敛更快，尤其是当tol高于1e-4时,可选cyclic,random	字符串	不必须	定数

[output]
train_predict	预测	训练集预测结果	一维数组(数值)
test_predict	预测	测试集预测结果	一维数组(数值)
train_score	正确率	训练集预测结果的正确率	数字
test_score	正确率	测试集预测结果的正确率	数字
intercept_	截距	决策特征中的独立术语	整数
coef_	参数向量	参数向量(成本函数公式中的W)。如果将1Dy传递给合适对象(非多任务使用)，则coef_是一维数组。注意coef_存储W，W.T的转置	二维数组
n_iter_	迭代次数	坐标下降求解器运行以达到指定公差的迭代次数	整数

[outline]


[describe]
以L1/L2混合范数为正则训练的多任务ElasticNet模型MultiTaskElasticNet的优化目标是：(1/(2*n_samples))*||Y-XW||_Fro^2+alpha*l1_ratio*||W||__21+0.5*alpha*(1-l1_ratio)*||W||_Fro^2哪里：：||W||_21=sum_isqrt(sum_jW_ij^2)即每行规范的总和
'''


def main(x_train, y_train, x_test, y_test,
         alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False, copy_X=True, max_iter=1000, tol=1e-4,
         warm_start=False, random_state=None, selection='cyclic'
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
    if type(normalize) is str:
        normalize = eval(normalize)
    if type(copy_X) is str:
        copy_X = eval(copy_X)
    if type(max_iter) is str:
        max_iter = eval(max_iter)
    if type(tol) is str:
        tol = eval(tol)
    if type(warm_start) is str:
        warm_start = eval(warm_start)
    if type(random_state) is str:
        random_state = eval(random_state)

    return r.run(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                 alpha=alpha,
                 l1_ratio=l1_ratio,
                 fit_intercept=fit_intercept,
                 normalize=normalize,
                 copy_X=copy_X,
                 max_iter=max_iter,
                 tol=tol,
                 warm_start=warm_start,
                 random_state=random_state,
                 selection=selection)


if __name__ == '__main__':
    import numpy as np
    import json
    array = np.loadtxt('D:\\123_2.csv', delimiter=',')
    array = np.array(array)
    array = array[0:20, :]
    y = array[:, -1]
    x = np.delete(array, -1, axis=1)
    y = np.vstack((y, y)).T
    back = main(x.tolist(), y, x.tolist(), y)
    print(back)
    for i in back:
        print(i + ":" + str(back[i]))

    json.dumps(back)

