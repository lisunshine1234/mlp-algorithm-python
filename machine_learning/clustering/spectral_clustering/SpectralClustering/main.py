import numpy as np
import run as  r

'''
[id]
144

[name]
SpectralClustering

[input]
x_train	训练集	训练集标签数据集	二维数组	必须	定数
y_train	测试集	测试集数据集	二维数组	必须	定数
n_clusters	簇数	默认为8,投影子空间的尺寸,可选整数,整数	字符串	不必须	定数
eigen_solver	eigen_solver	默认为None,使用特征值分解策略。 AMG需要安装pyamg。在非常大且稀疏的问题上，它可能会更快，但也可能导致不稳定,可选'lobpcg','amg','arpack'	字符串	不必须	定数
n_components	组件数	默认为None,用于频谱嵌入的本征向量数,可选整数,整数	字符串	不必须	定数
random_state	随机种子	默认为None,伪随机数生成器，用于在'eigen_solver=' amg'时通过K-Means初始化来分解lobpcg本征向量。使用int可以确定随机性,可选整数	整数	不必须	定数
n_init	随机初始化数量	默认为10,k均值算法将在不同质心种子下运行的次数。就惯性而言，最终结果将是n_init个连续运行的最佳输出,可选整数	整数	不必须	定数
gamma	gamma	默认为1.,rbf，poly，Sigmoid，laplacian和chi2内核的内核系数。忽略了'affinity=' nearest_neighbors,可选浮点数	浮点数	不必须	定数
affinity	亲和力	默认为rbf,如何构造亲和力矩阵。-'nearest_neighbors'：通过计算最近邻居的图来构造亲和矩阵。-'rbf'：使用径向基函数(RBF)内核构造亲和矩阵。-'precomputed'：将'X'解释为预先计算的亲和力矩阵。 -'precomputed_nearest_neighbors'：将'X'解释为预先计算的最近邻居的稀疏图，并通过选择'n_neighbors'最近邻居构建亲和力矩阵,可选,'rbf'	字符串	不必须	定数
n_neighbors	邻居数量	默认为10,使用最近邻居方法构造亲和力矩阵时要使用的邻居数量。忽略了'affinity=' rbf',可选整数,整数	字符串	不必须	定数
eigen_tol	eigen_tol	默认为0.0,当“ arpack”时，拉普拉斯矩阵特征分解的停止准则,可选浮点数	浮点数	不必须	定数
assign_labels	分配标签策略	默认为kmeans,用于在嵌入空间中分配标签的策略。拉普拉斯嵌入后，有两种分配标签的方法。可以应用k均值，它是一种流行的选择。但是它也可能对初始化敏感。离散化是另一种对随机初始化不太敏感的方法,可选'kmeans','discretize'	字符串	不必须	定数
degree	度	默认为3,多项式内核的度。被其他内核忽略,可选浮点数	浮点数	不必须	定数
coef0	coef0	默认为1,多项式和S形核的系数为零。被其他内核忽略,可选浮点数	浮点数	不必须	定数
kernel_params	kernel参数	默认为None,作为可调用对象传递的内核的参数(关键字参数)和值。被其他内核忽略,可选字符串,字符串,字典	字符串	不必须	定数
n_jobs	CPU数量	默认为None,要运行的并行作业数。 'None'表示1,可选整数	整数	不必须	定数

[output]
affinity_matrix_	亲和矩阵	用于聚类的亲和矩阵	二维数组
labels_	labels_	每个点的标签	一维数组

[outline]
将聚类应用于规范化拉普拉斯算子的投影。

[describe]
将聚类应用于规范化拉普拉斯算子的投影。
在实践中，当各个群集的结构高度不凸，或更普遍地说，当群集的中心和散布的度量值不适合完整群集时，频谱群集非常有用。
例如，当簇在2D平面上嵌套圆时。
如果亲和力是图的邻接矩阵，则可以使用此方法查找归一化图割。
当调用'fit'时，将使用任一核函数构造亲和矩阵，例如距离为'd(X，X)'的欧几里德的高斯(aka RBF)核:: np.exp(-gamma * d( X，X)** 2)或k最近邻居连接矩阵。
或者，使用'预先计算'，可以使用用户提供的亲和力矩阵。

'''


def main(x_train, y_train,
         n_clusters=8, eigen_solver=None, n_components=None, random_state=None, n_init=10, gamma=1., affinity='rbf', n_neighbors=10, eigen_tol=0.0,
         assign_labels='kmeans', degree=3, coef0=1, kernel_params=None, n_jobs=None
         ):
    if type(x_train) is str:
        x_train = eval(x_train)
    if type(y_train) is str:
        y_train = eval(y_train)
    if type(n_clusters) is str:
        n_clusters = eval(n_clusters)
    if type(n_components) is str:
        n_components = eval(n_components)
    if type(random_state) is str:
        random_state = eval(random_state)
    if type(n_init) is str:
        n_init = eval(n_init)
    if type(gamma) is str:
        gamma = eval(gamma)
    if type(n_neighbors) is str:
        n_neighbors = eval(n_neighbors)
    if type(eigen_tol) is str:
        eigen_tol = eval(eigen_tol)
    if type(degree) is str:
        degree = eval(degree)
    if type(coef0) is str:
        coef0 = eval(coef0)
    if type(kernel_params) is str:
        kernel_params = eval(kernel_params)
    if type(n_jobs) is str:
        n_jobs = eval(n_jobs)
    return r.run(x_train=x_train, y_train=y_train, n_clusters=n_clusters,
                 eigen_solver=eigen_solver,
                 n_components=n_components,
                 random_state=random_state,
                 n_init=n_init,
                 gamma=gamma,
                 affinity=affinity,
                 n_neighbors=n_neighbors,
                 eigen_tol=eigen_tol,
                 assign_labels=assign_labels,
                 degree=degree,
                 coef0=coef0,
                 kernel_params=kernel_params,
                 n_jobs=n_jobs)


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