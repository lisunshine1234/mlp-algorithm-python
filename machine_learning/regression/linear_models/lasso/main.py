import numpy as np
import run as r

'''
[id]
120

[name]
Lasso

[input]
x_train	训练集	训练集标签数据集	二维数组	必须	定数
y_train	测试集	测试集数据集	二维数组	必须	定数
x_test	训练集标签	训练集标签标签	一维数组	必须	定数
y_test	测试集标签	测试集标签	一维数组	必须	定数
alpha	alpha	默认为1.0,乘以L1项的常数。默认为1.0。alpha=0等价于一个普通的最小二乘，由LinearRegression对象求解。由于数值上的原因，使用alpha=0和Lasso对象是不建议的,可选浮点数	浮点数	不必须	定数
fit_intercept	计算截距	默认为True,是否计算该模型的截距。如果设置为False，将不会在计算中使用任何拦截。数据应该居中),可选整数,布尔值	字符串	不必须	定数
normalize	归一化	默认为False,当fit_intercept设置为False时，该参数将被忽略。如果为真，则回归量X将在回归前通过减去均值并除以l2-范数进行归一化。如果你想标准化，请使用preprocessing,可选布尔值	布尔值	不必须	定数
precompute	预先计算	默认为False,是否使用预先计算的克矩阵来加速计算。如果设置为自动让我们决定。克矩阵也可以作为参数传递。对于稀疏输入，这个选项总是True，以保持稀疏性,可选布尔值,数组	字符串	不必须	定数
copy_X	是否复制	默认为True,如果True，X将被复制;否则，它可能被覆盖,可选布尔值	布尔值	不必须	定数
max_iter	最大迭代次数	默认为1000,最大迭代次数,可选整数	整数	不必须	定数
tol	tol	默认为1e-4,优化的容忍度:如果更新小于tol，优化代码检查双间隙的最优性，并一直持续到它小于tol,可选浮点数	浮点数	不必须	定数
warm_start	warm_start	默认为False,当设置为True时，重用前面调用的解决方案以适合初始化，否则，就擦除前面的解决方案,可选布尔值	布尔值	不必须	定数
positive	positive	默认为False,当设置为True时，强制系数为正,可选布尔值	布尔值	不必须	定数
random_state	随机状态	默认为None,选择要更新的随机特性的伪随机数生成器的种子。当‘selection‘==‘random’时使用。在多个函数调用之间传递可重复输出的int,可选整数	整数	不必须	定数
selection	selection	默认为'cyclic',如果设置为随机，随机系数将在每次迭代时更新，而不是默认情况下按顺序遍历特征。这(设置为random)通常会导致明显更快的收敛速度，尤其是当tol高于1e-4时,可选cyclic,random	字符串	不必须	定数

[output]
train_predict	预测	训练集预测结果	一维数组(数值)
test_predict	预测	测试集预测结果	一维数组(数值)
train_score	正确率	训练集预测结果的正确率	数字
test_score	正确率	测试集预测结果的正确率	数字
coef_	参数向量	参数向量(成本函数公式中的w)	一维数组
sparse_coef_	稀疏参数向量	sparse_coef_是一个从coef_派生的只读属性	不定数组
intercept_	截距	决策函数中的独立项	整数
n_iter_	迭代次数	由坐标下降求解器运行以达到指定公差的迭代次数	整数


[outline]
线性模型先经过L1训练作为正则化器（又称为Lasso套索）

[describe]
线性模型先经过L1训练作为正则化器（又称为套索），
套索的优化目标是：（1	/	/（2	*	n_samples））*	||	y-Xw	||	^	2_2	+	alpha	*	||	w	||	_1
拉索模型正在以``l1_ratio	=	1.0''（没有L2损失）优化与Elastic	Net相同的目标函数。
'''


def main(x_train, y_train, x_test, y_test, alpha=1.0, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000, tol=1e-4,
   warm_start=False, positive=False, random_state=None, selection='cyclic'):
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
 if type(normalize) is str:
  normalize = eval(normalize)
 if type(precompute) is str and precompute != "auto":
  precompute = eval(precompute)
 if type(copy_X) is str:
  copy_X = eval(copy_X)
 if type(max_iter) is str:
  max_iter = eval(max_iter)
 if type(tol) is str:
  tol = eval(tol)
 if type(warm_start) is str:
  warm_start = eval(warm_start)
 if type(positive) is str:
  positive = eval(positive)
 if type(random_state) is str:
  random_state = eval(random_state)

 return r.run(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, alpha=alpha,
     fit_intercept=fit_intercept,
     normalize=normalize,
     precompute=precompute,
     copy_X=copy_X,
     max_iter=max_iter,
     tol=tol,
     warm_start=warm_start,
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
