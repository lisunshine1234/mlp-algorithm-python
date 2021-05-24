import numpy as np
import run as r

'''
[id]
88

[name]
chi2

[input]
x 数据集 数据集 二维数组 必须 定数
y 标签 标签 一维数组 必须 定数

[output]
pval F值 每个特征的p值 一维数组
chi2 p值 每个特征的chi2统计信息 一维数组

[outline]
计算每个非负特征与类之间的卡方统计量。

[describe]
计算每个非负特征与类之间的卡方统计量。
该分数可用于从X中选择测试卡方统计量具有最高值的n_features特征，该特征必须仅包含非负特征，例如布尔值或频率，相对于类。
回想一下，卡方检验可测量随机变量之间的相关性，因此使用此特征可以“淘汰”最有可能与类别无关，因此与分类无关的特征
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

