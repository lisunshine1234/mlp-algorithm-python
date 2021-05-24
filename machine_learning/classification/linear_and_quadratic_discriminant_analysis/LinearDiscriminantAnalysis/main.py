import numpy as np
import run as  r

'''
[id]
96

[name]
LinearDiscriminantAnalysis

[input]
x_train	训练集	训练集标签数据集	二维数组	必须	定数
y_train	测试集	测试集数据集	二维数组	必须	定数
x_test	训练集标签	训练集标签标签	一维数组	必须	定数
y_test	测试集标签	测试集标签	一维数组	必须	定数
solver	求解器	默认为svd,要使用的求解器，可能的值是：-'svd'：奇异值分解(默认值)。不计算协方差矩阵，因此建议将此求解器用于具有大量特征的数据。-'lsqr'：最小二乘解，可以与收缩组合。 -'eigen'：特征值分解，可以与收缩结合使用,可选'svd','lsqr','eigen','svd'	字符串	不必须	定数
shrinkage	收缩参数	默认为None,收缩参数，可能的值：-无：无收缩(默认)。-'auto'：使用Ledoit-Wolf引理自动收缩。-在0和1之间浮动：固定收缩参数。请注意，收缩仅适用于'lsqr'和'eigen'求解器,可选浮点数,'auto'	字符串	不必须	定数
priors	先验概率	默认为None,类的先验概率。默认情况下，从训练数据中推断出班级比例,可选数组	一维数组	不必须	定数
n_components	组件数	默认为None,用于降维的组件数(<= min(n_classes-1，n_features))。如果为None，则将设置为min(n_classes-1，n_features)。此参数仅影响'transform'方法,可选整数	整数	不必须	定数
store_covariance	存储协方差	默认为False,如果为True，则在求解器为'svd'时显式计算加权的类内协方差矩阵。矩阵始终为其他求解器计算和存储,可选布尔值	布尔值	不必须	定数
tol	tol	默认为1e-4,X的奇异值被认为是重要的绝对阈值，用于估计X的秩。丢弃奇异值不重要的维。仅在ifsolver为'svd'时使用,可选浮点数	浮点数	不必须	定数

[output]
train_predict	预测	训练集预测结果	一维数组(数值)
test_predict	预测	测试集预测结果	一维数组(数值)
train_score	正确率	训练集预测结果的正确率	数字
test_score	正确率	测试集预测结果的正确率	数字
coef_	参数向量	权重向量	一维数组
intercept_	截距	拦截项	一维数组
covariance_	加权类内协方差矩阵	加权类内协方差矩阵。它对应于'sum_k prior_k * C_k'，其中'C_k'是'k'类中样本的协方差矩阵。使用协方差的(潜在收缩)有偏估计量来估计'C_k'。如果求解器为'svd'，则仅在'store_covariance'为True时存在	二维数组
explained_variance_ratio_	成分方差百分比	每个选定成分说明的方差百分比。如果未设置'n_components'，则将存储所有成分，说明方差之和等于1.0。仅在使用特征svd解算器时可用	一维数组
means_	类别均值	类别均值	二维数组
priors_	类优先级	类优先级(总和为1)	一维数组
scalings_	缩放比例	类质心跨越的空间中特征的缩放比例，仅适用于'svd'和'eigen'求解器	二维数组
xbar_	总体均值	总体均值。仅在求解器为'svd'时存在	一维数组
classes_	类标签	唯一的类标签	一维数组

[outline]
线性判别分析 由拟合类生成的具有线性决策边界的分类器 数据的条件密度和使用贝叶斯规则

[describe]
线性判别分析 由拟合类生成的具有线性决策边界的分类器 数据的条件密度和使用贝叶斯规则
该模型将高斯密度拟合到每个类别，并假设所有类别 共享相同的协方差矩阵 拟合模型还可以用于减少输入的维数 通过将其投影到最有区别的方向，使用 'transform'方法

'''


def main(x_train, y_train, x_test, y_test, solver='svd', shrinkage=None, priors=None, n_components=None, store_covariance=False, tol=1e-4):
    if type(x_train) is str:
        x_train = eval(x_train)
    if type(y_train) is str:
        y_train = eval(y_train)
    if type(x_test) is str:
        x_test = eval(x_test)
    if type(y_test) is str:
        y_test = eval(y_test)
    if type(shrinkage) is str and shrinkage != "auto":
        shrinkage = eval(shrinkage)
    if type(priors) is str:
        priors = eval(priors)
    if type(n_components) is str:
        n_components = eval(n_components)
    if type(store_covariance) is str:
        store_covariance = eval(store_covariance)
    if type(tol) is str:
        tol = eval(tol)
    return r.run(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, solver=solver,
                 shrinkage=shrinkage,
                 priors=priors,
                 n_components=n_components,
                 store_covariance=store_covariance,
                 tol=tol)


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
