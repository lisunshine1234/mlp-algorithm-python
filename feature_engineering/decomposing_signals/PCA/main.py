import numpy as np
import run as r

'''
[id]
94

[name]
PCA

[input]
array 数据集 数据集 二维数组 必须 定数
label_index 标签列号 默认为None,表示所有列参与训练，输入整数代表标签所在的列号，可选整数 数字 不必须 定数
n_components n_components 默认为None,要保留的组件数。如果未设置n_components，则保留所有组件;n_components==min（n_samples，n_features）如果n_components==mle和svd_solver==full，则使用Minka的MLE来猜测尺寸。使用n_components==mle会将svd_solver==auto解释为svd_solver==full。如果0<n_components<1和svd_solver==full，请选择组件数，以使需要解释的方差量大于n_components指定的百分比。如果svd_solver==arpack，则组件数量必须严格小于n_features和n_samples的最小值。因此，无情况将导致：n_components==min（n_samples，n_features）-1,可选整数,浮点数,字符串 字符串 不必须 定数
copy copy 默认为True,如果为False，则传递给fit的数据将被覆盖，并且运行fit（X）.transform（X）将不会产生预期的结果，请改用fit_transform（X）,可选布尔值 布尔值 不必须 定数
whiten whiten 默认为False,如果为True（默认为False），则将components_向量乘以n_samples的平方根，然后除以奇异值，以确保不相关的输出具有单位分量方差。白化会从转换后的信号中删除一些信息（组件的相对方差标度），但有时可以通过使下游估算器的数据遵循某些硬性假设来提高下游估算器的预测准确性,可选布尔值 布尔值 不必须 定数
svd_solver svd_solver 默认为auto,auto：默认的策略是基于X.shape和n_components选择求解器的：如果输入数据大于500x500并且要提取的组件数小于数据最小维度的80％，然后启用更有效的“随机化”方法。否则，将计算出精确的完整SVD，然后选择截断;full：运行完全完整的SVD调用标准LAPACK解算器，并通过后处理选择组件;arpack：截断为n_components调用ARPACK解算器的SVD。严格地要求0<n_components<min（X.shape）;randomized：通过Halko等人的方法运行随机化的SVD,可选auto,full,arpack,randomized 字符串 不必须 定数
tol tol 默认为0.0,svd_solver==arpack计算的奇异值的公差,可选浮点数 浮点数 不必须 定数
iterated_power iterated_power 默认为auto,svd_solver==随机化计算出的幂方法的迭代次数,可选整数 整数 不必须 定数
random_state random_state 默认为None,在svd_solver==arpack或randomized时使用。在多个函数调用之间传递int以获得可重复的结果,可选整数 整数 不必须 定数

[output]
array 数组 训练之后的数组 二维数组
components_ 最大方差方向 特征空间中的主轴，表示数据中最大方差的方向。组件按``explained_variance_排序 二维数组
explained_variance_ 方差量 每个选定组件说明的方差量。等于X的协方差矩阵的n_components个最大特征值 一维数组
explained_variance_ratio_ 方差百分比 每个选定组件解释的方差百分比。如果未设置n_components，则将存储所有分量，并且比率之和等于1.0 一维数组
singular_values_ 组件奇异值 对应于每个所选组件的奇异值。奇异值等于低维空间中n_components变量的2范数 一维数组
mean_ 特征经验均值 根据训练集估算的每特征经验均值 一维数组
n_components_ 组件数 估计的组件数。当n_components设置为mle或0到1之间的数字（svd_solver==full）时，将从输入数据中估计该数字。否则，它等于参数n_components，如果n_components为None，则等于n_features和n_samples的较小值 整数
n_features_ 特征数量 训练数据中的特征数量 整数
n_samples_ 样本数 训练数据中的样本数 整数
noise_variance_ 噪声方差 根据Tipping-and-Bishop-1999的概率PCA模型估计的噪声协方差 整数

[outline]
主成分分析（PCA）。

[describe]
主成分分析（PCA）。
使用数据的奇异值分解的线性维数降低到它投影到低维空间。
 输入数据为中心，但应用SVD之前不进行缩放为每个特征。
它使用全SVD或通过Halko等人的方法中的随机截断SVD的LAPACK实现。
请注意，此类不支持稀疏输入。
'''


def main(array, label_index=None, n_components=None, copy=True, whiten=False, svd_solver='auto', tol=0.0,
         iterated_power='auto', random_state=None):
    if type(array) is str:
        array = eval(array)
    if type(label_index) is str:
        label_index = eval(label_index)
    if type(n_components) is str and n_components != "mle":
        n_components = eval(n_components)
    if type(copy) is str:
        copy = eval(copy)
    if type(whiten) is str:
        whiten = eval(whiten)
    if type(tol) is str:
        tol = eval(tol)
    if type(iterated_power) is str and iterated_power != "auto":
        iterated_power = eval(iterated_power)
    if type(random_state) is str:
        random_state = eval(random_state)
    return r.run(array=array,
                 label_index=label_index,
                 n_components=n_components,
                 copy=copy,
                 whiten=whiten,
                 svd_solver=svd_solver,
                 tol=tol,
                 iterated_power=iterated_power,
                 random_state=random_state)


if __name__ == '__main__':
    import numpy as np
    import json

    array = np.loadtxt('D:\\123_2.csv', delimiter=',')
    array = array[0:20, :]

    y = array[:, -1].tolist()
    x = np.delete(array, -1, axis=1).tolist()
    array = array.tolist()
    back = main(array, -1, n_components=2)

    print(back)
    for i in back:
        print(i + ":" + str(back[i]))

    json.dumps(back)
