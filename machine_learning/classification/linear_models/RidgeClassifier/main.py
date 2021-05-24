import numpy    as    np
import run    as    r

'''
[id]
100

[name]
RidgeClassifier

[input]
x_train	训练集	训练集标签数据集	二维数组	必须	定数
y_train	测试集	测试集数据集	二维数组	必须	定数
x_test	训练集标签	训练集标签标签	一维数组	必须	定数
y_test	测试集标签	测试集标签	一维数组	必须	定数
alpha	alpha	默认为1.0,正则强度；必须为正浮点数。正则化改善了问题的条件，并减少了估计的方差。较大的值表示更强的正则化。Alpha对应于其他线性模型中的1/(2C)。如果传递数组，则认为惩罚是特定于目标的。因此，它们必须在数量上对应,可选数组,浮点数	字符串	不必须	定数
fit_intercept	适合截距	默认为True,是否适合该模型的截距。如果设置为false，则不会在计算中使用截距(即X和y应该居中),可选整数,布尔值	字符串	不必须	定数
normalize	归一化	默认为False,当fit_intercept设置为False时，将忽略此参数。如果为True，则将在回归之前通过减去均值并除以l2-范数来对回归变量X进行归一化,可选布尔值	布尔值	不必须	定数
copy_X	是否复制	默认为True,如果为True，将复制X；否则为X。否则，它可能会被覆盖,可选布尔值	布尔值	不必须	定数
max_iter	最大迭代次数	默认为None,共轭梯度求解器的最大迭代次数。对于sparse_cg和lsqr求解器，默认值由scipy.sparse.linalg确定。对于下垂求解器，默认值为1000,可选整数	整数	不必须	定数
tol	精度	默认为0.001,解决方案的精度,可选浮点数	浮点数	不必须	定数
class_weight	class_weight	默认为None,与类别相关的权重，格式为{class_label：weight}。如果未给出，则所有类都应具有权重一。平衡模式使用y的值自动将权重与输入数据中的类频率成反比地调整为n_samples/(n_classes*np.bincount(y)),可选字典或balanced	字符串	不必须	定数
solver	解算器	默认为auto,-auto根据数据类型自动选择求解器。-svd使用X的奇异值分解来计算Ridge系数。对于奇异矩阵，比Cholesky更稳定。-cholesky通过获得封闭形式的解决方案。-sparse_cg使用共轭梯度求解器。作为一种迭代算法，对于大规模数据(可以设置tol和max_iter)，此求解器比cholsky更合适。-lsqr使用专用的正则化最小二乘例程。它是最快的，并且使用迭代过程。-sag使用随机平均梯度下降，saga使用其改进的无偏版本SAGA。两种方法都使用迭代过程，并且当n_samples和n_features都很大时，通常比其他求解器更快。当fit_intercept为True时，仅sag和sparse_cg支持稀疏输入	字符串	不必须	定数
random_state	随机状态	默认为None,在solver==sag或saga混洗数据时使用,可选整数	整数	不必须	定数

[output]
train_predict	预测	训练集预测结果	一维数组(数值)
test_predict	预测	测试集预测结果	一维数组(数值)
train_score	正确率	训练集预测结果的正确率	数字
test_score	正确率	测试集预测结果的正确率	数字
coef_	权重向量	权重向量	一维数组
intercept_	截距	决策特征中的独立术语。如果fit_intercept=False，则设置为0.0	整数
n_iter_	迭代次数	每个目标的实际迭代次数。仅适用于sag和lsqr求解器。其他求解器将返回None	一维数组
classes_	类标签	类标签	一维数组

[outline]
使用Ridge回归的分类器

[describe]
使用Ridge回归的分类器。
该分类器首先将目标值转换为“{-1，1}”，然后将问题视为回归任务（在多类情况下为多输出回归）。
'''

def main(x_train, y_train, x_test, y_test, alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=1e-3, class_weight=None,
         solver="auto", random_state=None):
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
    if type(copy_X) is str:
        copy_X = eval(copy_X)
    if type(max_iter) is str:
        max_iter = eval(max_iter)
    if type(tol) is str:
        tol = eval(tol)
    if type(class_weight) is str and class_weight != "balanced":
        class_weight = eval(class_weight)
    if type(random_state) is str:
        random_state = eval(random_state)

    return r.run(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, alpha=alpha,
                 fit_intercept=fit_intercept,
                 normalize=normalize,
                 copy_X=copy_X,
                 max_iter=max_iter,
                 tol=tol,
                 class_weight=class_weight,
                 solver=solver,
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
