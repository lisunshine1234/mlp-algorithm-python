import numpy as np
import run as r

'''
[id]
85

[name]
f_regression

[input]
x 数据集 数据集 二维数组 必须 定数
y 标签 标签 一维数组 必须 定数
center 居中 默认值为True，如果为true，则X和y将居中。 布尔值 不必须 定数

[output]
pval F值 特征F值 一维数组
F p值 F得分的p值 一维数组

[outline]
单变量线性回归测试。

[describe]
单变量线性回归测试。
线性模型测试每个许多回归量的个体效应。 这是一个特征选择的过程中要使用的计分函数，而不是一个独立的特征选择过程。
每个回归和目标之间的相关性被计算，即，（（X [:,Ⅰ] - 平均值（X [:, I]））*（Y - mean_y））/（STD（X [:, i]于）* STD（Y））。
它被转换成一个F得分然后到的p值
'''


def main(x, y, center=True):
    if type(x) is str:
        x = eval(x)
    if type(y) is str:
        y = eval(y)
    return r.run(x=x, y=y, center=center)


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

