import numpy as np
import run as  r

'''
[id]
97

[name]
QuadraticDiscriminantAnalysis

[input]
x_train	训练集	训练集标签数据集	二维数组	必须	定数
y_train	测试集	测试集数据集	二维数组	必须	定数
x_test	训练集标签	训练集标签标签	一维数组	必须	定数
y_test	测试集标签	测试集标签	一维数组	必须	定数
priors	先验概率	默认为None,班级先验。默认情况下，从训练数据中推断出班级比例,可选数组,数组	字符串	不必须	定数
reg_param	正则化	默认为0.,通过将S2转换为'S2 = (1 - reg_param) * S2 + reg_param * np.eye(n_features)'来对每个类别的协方差估计值进行正则化，其中S2对应于给定类别的'scaling_'属性,可选浮点数	浮点数	不必须	定数
store_covariance	存储协方差	默认为False,如果为True，则显式计算类协方差矩阵并将其存储在'self.covariance_'属性中,可选布尔值	布尔值	不必须	定数
tol	tol	默认为1.0e-4,奇异值被认为是重要的绝对阈值，用于估计'Xk'的秩，其中'Xk'是k类样本的居中矩阵。此参数不影响预测。它仅控制将特征视为共线时发出的警告,可选浮点数	浮点数	不必须	定数

[output]
train_predict	预测	训练集预测结果	一维数组(数值)
test_predict	预测	测试集预测结果	一维数组(数值)
train_score	正确率	训练集预测结果的正确率	数字
test_score	正确率	测试集预测结果的正确率	数字
covariance_	加权类内协方差矩阵	对于每个类别，给出使用该类别的样本估计的协方差矩阵。估计是无偏见的。仅在'store_covariance'为True时存在	二维数组
means_	means_	类平均	二维数组
priors_	类优先级	类优先级(总和为1)	一维数组
rotations_	旋转数	对于每个类别k，形状的数组(n_features，n_k)，其中'n_k = min(n_features, number of elements in class k)'V'是高斯分布的旋转，即其主轴。它对应于'Xk = U S Vt'，即来自'Xk'的SVD的特征向量矩阵，其中$是来自k类的样本的中心矩阵	二维数组
scalings_	缩放比例	对于每个类，包含沿其主轴的高斯分布的比例，即旋转坐标系中的方差。它对应于'S^2 /(n_samples - 1)'，其中'S'是'Xk'的SVD的奇异值的对角矩阵，其中'Xk'是k类的样本的中心矩阵	一维数组
classes_	类标签	唯一的类标签	一维数组

[outline]


[describe]
二次判别分析 生成具有二次决策边界的分类器 通过将类的条件密度拟合到数据 并使用贝叶斯定律 该模型将高斯密度拟合到每个类别

'''


def main(x_train, y_train, x_test, y_test,
         priors=None, reg_param=0., store_covariance=False, tol=1.0e-4
         ):
    if type(x_train) is str:
        x_train = eval(x_train)
    if type(y_train) is str:
        y_train = eval(y_train)
    if type(x_test) is str:
        x_test = eval(x_test)
    if type(y_test) is str:
        y_test = eval(y_test)
    if type(priors) is str:
        priors = eval(priors)
    if type(reg_param) is str:
        reg_param = eval(reg_param)
    if type(store_covariance) is str:
        store_covariance = eval(store_covariance)
    if type(tol) is str:
        tol = eval(tol)
    return r.run(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, priors=priors,
                 reg_param=reg_param,
                 store_covariance=store_covariance,
                 tol=tol
                 )


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