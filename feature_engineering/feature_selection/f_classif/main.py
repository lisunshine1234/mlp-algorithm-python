import numpy as np
import run as r

'''
[id]
87

[name]
f_classif

[input]
x 数据集 数据集 二维数组 必须 定数
y 标签 标签 一维数组 必须 定数

[output]
pval F值 F值的集合 一维数组
F p值 p值的集合 一维数组

[outline]
计算提供的样本的方差分析f值。

[describe]
计算提供的样本的方差分析f值。
'''


def main(x, y):
    if type(x) is str:
        x = eval(x)
    if type(y) is str:
        y = eval(y)
    return r.run(x=x, y=y)


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

