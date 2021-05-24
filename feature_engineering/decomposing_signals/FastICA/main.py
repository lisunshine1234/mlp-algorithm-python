import numpy as np
import run as r
'''
[id]
91

[name]
FastICA

[input]
array 数据集 数据集 二维数组 必须 定数
label_index 标签列号 默认为None,表示所有列参与训练，输入整数代表标签所在的列号，可选整数 数字 不必须 定数
n_components 组件数 默认为None,要使用的组件数。如果没有通过，则全部使用，可选整数 数字 不必须 定数
algorithm 算法 默认为parallel,对FastICA应用并行或整型算法，可选'parallel','deflation' 字符串 不必须 定数
whiten 控制过程的详细程度 默认为True,如果whiten为false，则数据已经被认为是whitening，并且不执行whitening。 布尔值 不必须 定数
fun 函数 默认为logcosh,G函数的函数形式，用于近似负熵。可以是“logcosh”，“exp”或“cube”，可选'logcosh','exp','cube' 字符串 不必须 定数
fun_args 函数参数 默认为None,要发送给函数形式的参数。如果为空，并且如果fun='logcosh'，则fun_args将采用值{'alpha'：1.0} json 不必须 定数
max_iter 最大迭代次数 默认为200,拟合期间的最大迭代次数，可选整数 整数 不必须 定数
tol 数值误差公差 默认为0.0001,每次迭代的更新公差，可选浮点数 浮点数 不必须 定数
dict_init 初始化算法混合矩阵 默认为None,用于初始化算法的混合矩阵，可选二维数组(n_components,n_components) 二维数组 不必须 定数
random_state 随机状态 默认为None,未指定时用于初始化w_init，具有正态分布。传递一个int，以在多个函数调用之间获得可重现的结果，可选整数 整数 不必须 定数

[output]
array 数组 训练之后的数组 二维数组
components_ 组件 线性运算符应用于数据以获取独立源。当whiten为False时，它等于分解矩阵；当whiten为True时,等于矩阵相乘 二维数组
mean_ 均值超过特征 均值超过特征。仅在`self.whiten`为True时设置 一维数组
n_iter_ 迭代次数 如果算法为“deflation”，则n_iter是所有组件之间运行的最大迭代次数。否则，它们只是收敛的迭代次数 数字
whitening_ 漂白 仅在漂白为“True”时设置。这是将数据投影到第一个n_components主成分上的美白前矩阵。 二维数组

[outline]
独立分量分析的快速算法

[describe]
独立分量分析的快速算法
'''

def main(array, label_index=None, n_components=None, algorithm='parallel', whiten=True,
         fun='logcosh', fun_args=None, max_iter=200, tol=1e-4,
         w_init=None, random_state=None):
    if type(array) is str:
        array = eval(array)
    if type(label_index) is str:
        label_index = eval(label_index)
    if type(n_components) is str:
        n_components = eval(n_components)
    if type(whiten) is str:
        whiten = eval(whiten)
    if type(fun_args) is str:
        fun_args = eval(fun_args)
    if type(max_iter) is str:
        max_iter = eval(max_iter)
    if type(tol) is str:
        tol = eval(tol)
    if type(w_init) is str:
        w_init = eval(w_init)
    if type(random_state) is str:
        random_state = eval(random_state)
    return r.run(array=array,
                 label_index=label_index,
                 n_components=n_components,
                 algorithm=algorithm,
                 whiten=whiten,
                 fun=fun,
                 fun_args=fun_args,
                 max_iter=max_iter,
                 tol=tol,
                 w_init=w_init,
                 random_state=random_state)


if __name__ =='__main__':
    array = np.loadtxt('D:\\123_2.csv', delimiter=',')
    y = array[:, -1]
    x = np.delete(array, -1, axis=1)

    x = x.tolist()
    y = y.tolist()
    array = array.tolist()
    print(main(array, -1, n_components=2))
