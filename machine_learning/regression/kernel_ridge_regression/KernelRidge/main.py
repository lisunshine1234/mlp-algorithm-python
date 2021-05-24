import numpy as np
import run as r

'''
[id]
113

[name]
KernelRidge

[input]
x_train	训练集	训练集标签数据集	二维数组	必须	定数
y_train	测试集	测试集数据集	二维数组	必须	定数
x_test	训练集标签	训练集标签标签	一维数组	必须	定数
y_test	测试集标签	测试集标签	一维数组	必须	定数
alpha	alpha	默认为1,正则强度；必须为正浮点数。正则化改善了问题的条件并减小了估计的方差。较大的值表示更强的正则化。Alpha对应于其他线性模型中的'1 / (2C)'，如果通过数组，则认为特定于目标的处罚区域。因此，它们必须在数量上对应,可选数组,浮点数	字符串	不必须	定数
kernel	内核	默认为'linear',内部使用的内核映射。如果'kernel'是'precomputed'，则假定X是内核矩阵,可选,"linear", "additive_chi2", "chi2", "poly", "polynomial", "rbf","laplacian", "sigmoid", "cosine"	字符串	不必须	定数
gamma	gamma	默认为None,RBF，laplacian，polynomial，exponential chi2 和 sigmoid kernels的Gamma参数。被其他内核忽略,可选浮点数	浮点数	不必须	定数
degree	内核的度	默认为3,多项式内核的度。被其他内核忽略,可选浮点数	浮点数	不必须	定数
coef0	coef0	默认为1,多项式和sigmoid内核的零系数，被其他内核忽略,可选浮点数	浮点数	不必须	定数
kernel_params	kernel_params	默认为None,内核函数作为可调用对象传递的附加参数(关键字参数),可选字符串,字符串	字符串	不必须	定数

[output]
train_predict	预测	训练集预测结果	一维数组(数值)
test_predict	预测	测试集预测结果	一维数组(数值)
train_score	正确率	训练集预测结果的正确率	数字
test_score	正确率	测试集预测结果的正确率	数字
dual_coef_	权重向量	核空间中权重向量的表示	一维数组
X_fit_	X_fit_	训练数据，这也是预测所必需的。如果kernel == 'precomputed'，则它是预先计算的训练矩阵，形状为(n_samples，n_samples)	二维数组

[outline]
内核岭回归(KRR)结合了岭回归(线性最小 平方和带有l2-范数正则化的平方)和内核技巧

[describe]
内核岭回归
内核岭回归(KRR)结合了岭回归(线性最小 平方和带有l2-范数正则化的平方)和内核技巧
因此 学习由相应内核诱导的空间中的线性函数，并 数据
对于非线性内核，这对应于非线性 在原始空间中发挥作用
KRR学习的模型的形式与支持向量相同 回归(SVR)
但是，使用了不同的损失函数：KRR使用 支持向量回归使用对epsilon不敏感的平方误差损失 损失，同时结合l2正则化
与SVR相比， KRR模型可以封闭形式完成，通常对于 中型数据集
另一方面，学习的模型是非稀疏的 因此比SVR慢，后者在以下情况下学习稀疏模型，即epsilon> 0 预测时间
此估算器对多变量回归具有内置支持 (即，当y是形状为[n_samples，n_targets]的二维数组时)
'''


def main(x_train, y_train, x_test, y_test, alpha=1, kernel="linear", gamma=None, degree=3, coef0=1, kernel_params=None):
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
    if type(gamma) is str:
        gamma = eval(gamma)
    if type(degree) is str:
        degree = eval(degree)
    if type(coef0) is str:
        coef0 = eval(coef0)
    if type(kernel_params) is str:
        kernel_params = eval(kernel_params)

    return r.run(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, alpha=alpha,
                 kernel=kernel,
                 gamma=gamma,
                 degree=degree,
                 coef0=coef0,
                 kernel_params=kernel_params)

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
