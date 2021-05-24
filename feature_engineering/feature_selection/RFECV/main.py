import numpy as np
import run as r

'''
[id]
80

[name]
RFECV

[input]
x 数据集 数据集 二维数组 必须 定数
y 标签 标签 一维数组 必须 定数
estimator 基本估算器 用来构建变压器的基本估算器。可选见算法标签详情 字符串 必须 定数
step 步数 默认为1，如果大于或等于1，则`step'对应于每次迭代要删除的个特征。如果在（0.0，1.0）之内，则`step'对应于每次迭代要删除的特征的百分比（四舍五入）。请注意，最后一次迭代可能会删除少于step个特征，以便达到“min_features_to_select”。 数字 不必须 定数
min_features_to_select 最少选择特征数 默认为1，最少选择特征数，可选整数 数字 不必须 定数
cv 交叉验证拆分策略 默认为5，确定交叉验证拆分策略，可选整数 数字 不必须 定数
scoring 评分器 默认为None,带有签名`scorer（estimator，X，y）'的字符串或评分器可调用对象/函数 字符串 不必须 定数
verbose 最高p值 默认为0，控制输出的详细程度，可选浮点数 数字 不必须 定数

[output]
n_features_  特征数量 具有交叉验证的所选特征的数量 数字
ranking_ 特征排名 特征排名，使得“ ranking_ [i]”对应于第i个特征的排名位置。选定的（即最佳估计）特征被分配为等级1。 一维数组
pvalues_support_ 特征选择的掩码  特征选择的掩码 一维数组

[outline]
通过消除递归特征和交叉验证最佳特征数选择来进行特征排名

[describe]
通过消除递归特征和交叉验证最佳特征数选择来进行特征排名
estimator可选：
'SVR'
'SVC'
'NuSVR'
'OneClassSVM'
'ARDRegression'
'BayesianRidge'
'ElasticNet'
'ElasticNetCV'
'HuberRegressor'
'Lars'
'LarsCV'
'Lasso'
'LassoCV'
'LassoLars'
'LassoLarsCV'
'LassoLarsIC'
'LinearRegression'
'OrthogonalMatchingPursuit'
'OrthogonalMatchingPursuitCV'
'PassiveAggressiveClassifier'
'PassiveAggressiveRegressor'
'Perceptron'
'Ridge'
'RidgeClassifier'
'RidgeClassifierCV'
'RidgeCV'
'SGDClassifier'
'SGDRegressor'
'TheilSenRegressor'
'LinearDiscriminantAnalysis'
'DecisionTreeClassifier'
'DecisionTreeRegressor'
'ExtraTreeClassifier'
'ExtraTreeRegressor'
'''
def main(x, y, estimator, step=1, min_features_to_select=1, cv=None,
         scoring=None, verbose=0):

    if type(x) is str:
        x = eval(x)
    if type(y) is str:
        y = eval(y)
    if type(min_features_to_select) is str:
        min_features_to_select = eval(min_features_to_select)
    if type(step) is str:
        step = eval(step)
    if type(cv) is str:
        cv = eval(cv)
    if type(verbose) is str:
        verbose = eval(verbose)
    return r.run(x=x, y=y, estimator=estimator, step=step, min_features_to_select=min_features_to_select, cv=cv,
                 scoring=scoring, verbose=verbose)


if __name__ == '__main__':
    import numpy as np
    import json

    array = np.loadtxt('D:\\123_2.csv', delimiter=',')
    array = array[0:20, :]

    y = array[:, -1].tolist()
    x = np.delete(array, -1, axis=1).tolist()
    array = array.tolist()
    back = main(x, y,'SVC')

    print(back)
    for i in back:
        print(i + ":" + str(back[i]))

    json.dumps(back)
