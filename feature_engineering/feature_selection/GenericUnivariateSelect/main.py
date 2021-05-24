import numpy as np
import run as r

'''
[id]
84

[name]
GenericUnivariateSelect

[name]
SelectFpr

[input]
x 数据集 数据集 二维数组 必须 定数
y 标签 标签 一维数组 必须 定数
score_func 分数函数 默认值为f_class，函数接受两个数组X和y，并返回一对数组（分数，pvalue）或带分数的单个数组。可选见算法标签详细描述 字符串 不必须 定数
mode 最高p值 默认为percentile，特征选择模式，可选'percentile', 'k_best', 'fpr', 'fdr', 'fwe' 字符串 不必须 定数
param 最高p值 0.00001，对应模式的参数，可选浮点数或整数 数字 不必须 定数

[output]
scores_ 特征分数 特征分数 一维数组
pvalues_ p值 特征分数的p值 一维数组

[outline]
单变量特征选择与配置策略

[describe]
单变量特征选择与配置策略
score_func：f_classif、mutual_info_classif、chi2、f_regression、mutual_info_regression、SelectPercentile、SelectKBest、SelectFpr、SelectFdr、SelectFwe
'''


def main(x, y, score_func='f_classif', mode='percentile', param=1e-5):
    if type(x) is str:
        x = eval(x)
    if type(y) is str:
        y = eval(y)
    if type(param) is str:
        param = eval(param)
    return r.run(x=x, y=y, score_func=score_func, mode=mode, param=param)


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

