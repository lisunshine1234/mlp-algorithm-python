import numpy as np
import run as r

'''
[id]
78

[name]
SelectFpr

[input]
x 数据集 数据集 二维数组 必须 定数
y 标签 标签 一维数组 必须 定数
score_func 分数函数 默认值为f_class，函数接受两个数组X和y，并返回一对数组（分数，pvalue）或带分数的单个数组。可选f_classif、chi2、f_regression 字符串 不必须 定数
alpha 最高p值 0.05，要保留的特征的最高p值，可选浮点数 数字 不必须 定数

[output]
scores_ 特征分数 特征分数 一维数组
pvalues_ p值 特征分数的p值 一维数组
transform 数组 特征选择之后的数组 二维数组

[outline]
根据FPR测试，在alpha以下选择p值。

[describe]
根据FPR测试，在alpha以下选择p值。
FPR测试代表误报率测试。它控制错误检测的总量。
'''
def main(x, y, score_func='f_classif', alpha=5e-2):

    if type(x) is str:
        x = eval(x)
    if type(y) is str:
        y = eval(y)
    if type(alpha) is str:
        alpha = eval(alpha)
    return r.run(x=x, y=y, score_func=score_func, alpha=alpha)


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