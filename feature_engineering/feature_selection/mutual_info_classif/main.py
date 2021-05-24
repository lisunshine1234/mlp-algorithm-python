import numpy as np
import run as r

'''
[id]
83

[name]
mutual_info_classif

[input]
x 数据集 数据集 二维数组 必须 定数
y 标签 标签 一维数组 必须 定数
discrete_features 离散特征  默认为auto,如果是布尔值，则确定是考虑所有特征是离散特征还是连续特征。如果是数组，则它应该是具有形状（n_features）的布尔蒙版，或者是具有离散特征索引的数组。如果为“ auto”，则对于密集的“ X”将其分配为False，对于稀疏的“ X”将其分配为True。可选’auto‘、布尔、数组 字符串 不必须 定数
n_neighbors 邻居数 默认为3，用于连续变量的MI估计的邻居数。较高的值会减少估计的方差，但可能会带来偏差。可选整数 数字 不必须 定数
copy 是否复制 默认为True，是否复制给定的数据。如果设置为False，则初始数据将被覆盖。 布尔值 不必须 定数
random_state 随机数 默认为None，确定随机数生成，以将小噪声添加到连续变量中以删除重复值。在多个函数调用之间传递int以获得可重复的结果，可选整数 数字 不必须 定数

[output]
mi  互信息 具有交叉验证的所选特征的数量 数字

[outline]
估计离散目标变量的互信息。

[describe]
估计离散目标变量的互信息。
两个随机变量之间的互信息（MI）_是一个非负值，用于衡量变量之间的相关性。
当且仅当两个随机变量是独立的，并且等于较高的值意味着较高的依赖性时，它等于零。
该函数依赖于非参数方法，该方法基于[k]最近邻居距离的熵估计。
'''
def main(x, y, discrete_features='auto', n_neighbors=3, copy=True, random_state=None):

    if type(x) is str:
        x = eval(x)
    if type(y) is str:
        y = eval(y)
    if type(discrete_features) is str and discrete_features != 'auto':
        discrete_features = eval(discrete_features)
    if type(n_neighbors) is str:
        n_neighbors = eval(n_neighbors)
    if type(copy) is str:
        copy = eval(copy)
    if type(random_state) is str:
        random_state = eval(random_state)
    return r.run(x=x, y=y, discrete_features=discrete_features, n_neighbors=n_neighbors, copy=copy,
                 random_state=random_state)


if __name__ == '__main__':
    import numpy as np
    import json
    array = np.loadtxt('D:\\123_2.csv', delimiter=',')

    array = np.array(array)
    y = array[:, -1]
    x = np.delete(array, -1, axis=1)
    a = [1, 2, 3, 1, 2, 3]
    b = '[True, True, False, True, False, True, True]'
    print(main(x.tolist(), y, discrete_features=b))
    print(main(x.tolist(), y, discrete_features=a))
