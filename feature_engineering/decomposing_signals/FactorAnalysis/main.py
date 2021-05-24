import numpy as np
import run as r

'''
[id]
90

[name]
FactorAnalysis

[input]
array 数据集 数据集 二维数组 必须 定数
label_index 标签列号 默认为None,表示所有列参与训练，输入整数代表标签所在的列号，可选整数 数字 不必须 定数
n_components 组件数 默认为None,潜在空间的维数，是在“变换”之后获得的“X”的分量数。如果为None，则将n_components设置为特征数量，可选整数 数字 不必须 定数
tol 数值误差公差 默认为0.01,停止对数可能性的容忍度，可选浮点数 浮点数 不必须 定数
copy 是否复制 默认为True,是否复制X。如果为“False”，则在拟合期间输入X被覆盖 布尔值 不必须 定数
max_iter 最大迭代次数 默认为1000,最大迭代次数，可选整数 整数 不必须 定数
noise_variance_init 噪声方差 默认为None,每个特征的噪声方差的初始猜测。如果为None，则默认为长度n_features的一矩阵，可选一维数组 一维数组 不必须 定数
svd_method SVD方法 默认为randomized,使用哪种SVD方法。如果'lapack'使用scipy.linalg的标准SVD，如果'randomized'使用快速的“randomized_svd”函数。默认为“随机化”。对于大多数应用程序，“随机化”将足够精确，同时可显着提高速度。通过为“iterated_power”设置更高的值也可以提高精度。如果这还不够，为了获得最大的精度，您应该选择“lapack” 字符串 不必须 定数
iterated_power 稀疏控制参数 默认为3,幂方法的迭代次数。默认为3。仅在``svd_method''等于'randomized'时使用，可选整数 整数 不必须 定数
random_state 随机状态 默认为0,仅在``svd_method''等于'randomized'时使用。在多个函数调用之间传递int以获得可重复的结果，可选整数 整数 不必须 定数

[output]
array 数组 训练之后的数组 二维数组
components_ 差异最大的组件 差异最大的组件 二维数组
loglike_ 似然率 每次迭代的对数似然率。 一维数组
noise_variance_ 估计噪声方差 每个特征的估计噪声方差 一维数组
n_iter_ 迭代次数 迭代次数 数字
mean_ 经验均值 根据训练集估算的每特征经验均值。 二维数组

[outline]
因子分析

[describe]
因子分析（FA）
线性高斯模型生成一个简单的潜在变量。
假定观察到通过的低维潜在因子和增加的高斯噪声的线性变换所引起的。
不失一般性的因素是根据具有零均值和协方差单元的高斯分布。
噪声也是零均值和具有任意的对角协方差矩阵。
如果我们将进一步限制模型，通过假定高斯噪声甚至各向同性（所有对角线项是相同的），我们将获得PPCA。
FactorAnalysis执行所谓的加载矩阵，潜变量观察到的那些的变换的最大似然估计，使用基于SVD的方法.
'''

def main(array, label_index=None, n_components=None, tol=1e-2, copy=True,
         max_iter=1000, noise_variance_init=None, svd_method='randomized',
         iterated_power=3, random_state=0):
    if type(array) is str:
        array = eval(array)
    if type(label_index) is str:
        label_index = eval(label_index)
    if type(n_components) is str:
        n_components = eval(n_components)
    if type(tol) is str:
        tol = eval(tol)
    if type(copy) is str:
        copy = eval(copy)
    if type(max_iter) is str:
        max_iter = eval(max_iter)
    if type(noise_variance_init) is str:
        noise_variance_init = eval(noise_variance_init)
    if type(iterated_power) is str:
        iterated_power = eval(iterated_power)
    if type(random_state) is str:
        random_state = eval(random_state)

    return r.run(array=array,
                 label_index=label_index,
                 n_components=n_components,
                 tol=tol,
                 copy=copy,
                 max_iter=max_iter,
                 noise_variance_init=noise_variance_init,
                 svd_method=svd_method,
                 iterated_power=iterated_power,
                 random_state=random_state)


if __name__ == '__main__':
    import numpy as np
    import json
    array = np.loadtxt('D:\\123_2.csv', delimiter=',')
    y = array[:, -1]
    x = np.delete(array, -1, axis=1)

    x = x.tolist()
    y = y.tolist()
    array = array.tolist()
    print(main(array, -1, n_components=2))
