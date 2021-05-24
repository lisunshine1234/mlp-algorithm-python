import numpy as np
import run as r
'''
[id]
92

[name]
IncrementalPCA

[input]
array 数据集 数据集 二维数组 必须 定数
label_index 标签列号 默认为None,表示所有列参与训练，输入整数代表标签所在的列号，可选整数 数字 不必须 定数
n_components 组件数 默认为None,要保留的组件数。如果n_components为None，则n_components设置为min（n_samples，n_features），可选整数 数字 不必须 定数
whiten 是否复制 默认为True,如果为True（默认情况下为False），components_向量将被n_samples乘以components_来确保不相关的输出具有单位逐项方差。泛白会从转换后的信号中删除一些信息（组件的相对方差标度），但有时可以通过使数据遵守某些严格的假设来提高下游估算器的预测精度 布尔值 不必须 定数
copy 是否复制 默认为True,如果为False，则X将被覆盖。`copy=False`可以用来节省内存，但是一般使用是不安全的。 布尔值 不必须 定数
batch_size 批次 默认为None,每个批次要使用的样本数。如果'batch_size'为'None'，则从数据中推断'batch_size'并将其设置为'5*n_features'，以在近似精度和内存消耗之间取得平衡。可选整数 整数 不必须 定数

[output]
array 数组 训练之后的数组 二维数组
components_ 差异最大的组件 差异最大的组件 二维数组
explained_variance_ 组件差异 每个选定组件所解释的差异。 一维数组
explained_variance_ratio_ 组件方差百分比 每个选定组件解释的方差百分比。如果存储了所有分量，则解释的方差之和等于1.0 一维数组
singular_values_ 组件奇异值 对应于每个所选组件的奇异值。奇异值等于低维空间中n_components变量的2范数。 一维数组
mean_ 经验均值 按特征的经验均值。 一维数组
var_ 经验方差 每个特征的经验方差。 一维数组
noise_variance_ 噪声协方差 根据Tipping-and-Bishop-1999的概率PCA模型估计的噪声协方差。 数字
n_components_ 估计组件数 估计的组件数。 数字
n_samples_seen_ 估计器处理的样本数 估计器处理的样本数。 数字
batch_size_ 批次大小 'batch_size'推断批次大小。 数字

[outline]
增量主成分分析（IPCA）。

[describe]
增量主成分分析（IPCA）。
使用数据的奇异值分解，只保留最显著奇异向量到数据投影到较低维空间中的线性维数降低。
输入数据为中心，但应用SVD之前不进行缩放为每个特征。
根据输入数据的大小，该算法可以更多的内存比PCA有效，并允许稀疏输入。
该算法不断内存的复杂性，顺序上batch_size*n_features ，能够使用的np.memmap文件，而无需将整个文件加载到内存中。
为稀疏矩阵，所述输入被转换为密分批（为了能够减去均值），其避免了在任一时间存储整个稠密矩阵。
每个SVD的计算开销是O(batch_size*n_features ** 2)但只有2 *样品的batch_size在时间内保持在存储器中。
会有n_samples / batch_size SVD计算得到的主成分，相对于1个大SVD复杂的O(n_samples*n_features ** 2)为PCA。
'''

def main(array, label_index=None, n_components=None, whiten=False, copy=True, batch_size=None):
    if type(array) is str:
        array = eval(array)
    if type(label_index) is str:
        label_index = eval(label_index)
    if type(n_components) is str:
        n_components = eval(n_components)
    if type(whiten) is str:
        whiten = eval(whiten)
    if type(copy) is str:
        copy = eval(copy)
    if type(batch_size) is str:
        batch_size = eval(batch_size)
    return r.run(array=array,
                 label_index=label_index,
                 n_components=n_components,
                 whiten=whiten,
                 copy=copy,
                 batch_size=batch_size)


if __name__ == '__main__':
    import numpy as np
    import json

    array = np.loadtxt('D:\\123_2.csv', delimiter=',')
    array = array[0:20, :]

    y = array[:, -1].tolist()
    x = np.delete(array, -1, axis=1).tolist()
    array = array.tolist()
    back = main(array)

    print(back)
    for i in back:
        print(i + ":" + str(back[i]))

    json.dumps(back)