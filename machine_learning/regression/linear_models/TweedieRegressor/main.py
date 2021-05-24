import numpy as np
import run as  r

'''
[id]
118

[name]
TweedieRegressor

[input]
x_train	训练集	训练集标签数据集	二维数组	必须	定数
y_train	测试集	测试集数据集	二维数组	必须	定数
x_test	训练集标签	训练集标签标签	一维数组	必须	定数
y_test	测试集标签	测试集标签	一维数组	必须	定数
power	power	默认为0,幂决定了潜在的目标分布.Normal=0，Poisson=1，Compound-Poisson-Gamma=(1,2)，Gamma=2，Inverse-Gaussian=3。For0<power<1，不存在分布,可选浮点数	浮点数	不必须	定数
alpha	alpha	默认为1,该常数乘以罚分项，从而确定正则化强度。alpha=0等同于未惩罚的GLM。在这种情况下，设计矩阵X必须具有完整的列等级(无共线性),可选浮点数	浮点数	不必须	定数
link	link	默认为'auto',GLM的链接函数，即从线性预测变量X@coeff+intercept映射到预测y_pred。选项自动根据所选族来设置链接，如下所示：-正态分布的身份-泊松，伽玛和高斯逆分布的对数,可选auto,identity,log	字符串	不必须	定数
fit_intercept	计算截距	默认为True,指定是否应将常数(也称为偏差或截距)添加到线性预测变量(X@coef+截距),可选整数,布尔值	字符串	不必须	定数
max_iter	最大迭代次数	默认为100,求解器的最大迭代次数,可选整数	整数	不必须	定数
tol	tol	默认为1e-4,停止标准。对于lbfgs求解器，当max{|g_j|，j=1，...，d}<=tol时，迭代将停止，其中g_j是梯度的第j个分量(导数))的目标函数,可选浮点数	浮点数	不必须	定数
warm_start	warm_start	默认为False,如果设置为True，请重用上一次对fit的调用的解决方案，作为coef_和intercept_的初始化,可选布尔值	布尔值	不必须	定数
verbose	详细程度	默认为0,对于lbfgs求解器，请将verbose设置为任何正数以表示详细程度,可选整数	整数	不必须	定数

[output]
train_predict	预测	训练集预测结果	一维数组(数值)
test_predict	预测	测试集预测结果	一维数组(数值)
train_score	正确率	训练集预测结果的正确率	数字
test_score	正确率	测试集预测结果的正确率	数字
coef_	参数向量	GLM中线性预测变量(X@coef_+intercept_)的估计系数	一维数组
intercept_	截距	拦截(又称偏差)已添加到线性预测变量中	整数
n_iter_	迭代次数	求解器中使用的实际迭代数	整数

[outline]


[describe]
Logistic回归(又名logit，MaxEnt)分类器。
在多类情况下，训练算法使用一比休息(OvR)如果multi_class选项设置为ovr，并使用如果multi_class选项设置为multinomial，则交叉熵损失。
(当前，多项式选项仅受lbfgs支持，sag，saga和newton-cg解算器。
)此类使用liblinear库，newton-cg，sag，saga和lbfgs求解器。
**注意默认情况下应用正则化**。
它既可以处理密集和稀疏输入。
使用包含64位的C排序数组或CSR矩阵浮动以获得最佳性能；任何其他输入格式将被转换(并复制)。
newton-cg，sag和lbfgs求解器仅支持L2正则化具有原始公式，或者没有正则化。
liblinear求解器支持L1和L2正则化，仅对L2罚款。
Elastic-Net正则化仅受saga求解器
'''


def main(x_train, y_train, x_test, y_test,
         power=0.0, alpha=1.0, fit_intercept=True, link='auto', max_iter=100, tol=1e-4, warm_start=False, verbose=0
         ):
    if type(x_train) is str:
        x_train = eval(x_train)
    if type(y_train) is str:
        y_train = eval(y_train)
    if type(x_test) is str:
        x_test = eval(x_test)
    if type(y_test) is str:
        y_test = eval(y_test)
    if type(power) is str:
        power = eval(power)
    if type(alpha) is str:
        alpha = eval(alpha)
    if type(fit_intercept) is str:
        fit_intercept = eval(fit_intercept)
    if type(max_iter) is str:
        max_iter = eval(max_iter)
    if type(tol) is str:
        tol = eval(tol)
    if type(warm_start) is str:
        warm_start = eval(warm_start)
    if type(verbose) is str:
        verbose = eval(verbose)

    return r.run(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, power=power,
                 alpha=alpha,
                 fit_intercept=fit_intercept,
                 link=link,
                 max_iter=max_iter,
                 tol=tol,
                 warm_start=warm_start,
                 verbose=verbose)


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
