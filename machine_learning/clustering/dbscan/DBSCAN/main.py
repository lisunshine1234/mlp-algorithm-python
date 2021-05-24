import numpy as np
import run as  r

'''
[id]
146

[name]
DBSCAN

[input]
x_train	训练集	训练集标签数据集	二维数组	必须	定数
y_train	测试集	测试集数据集	二维数组	必须	定数
eps	eps	默认为0.5,一个样本的两个样本之间的最大距离应视为另一个样本的邻域。这不是群集中点的距离的最大界限。这是为数据集和距离函数适当选择的最重要的DBSCAN参数,可选浮点数	浮点数	不必须	定数
min_samples	min_samples	默认为5,一个点被视为核心点的邻域中的样本数量(或总重量)。这包括点本身,可选整数	整数	不必须	定数
metric	度量	默认为euclidean,计算要素阵列中实例之间的距离时使用的度量。如果metric是字符串或可调用，则它必须是：func：'sklearn.metrics.pairwise_distances'为其metric参数所允许的选项之一。如果度量为'precomputed'，则假定X为距离矩阵，并且必须为正方形。 X可以是：term：'Glossary <sparse graph>'，在这种情况下，只有'nonzero'元素可以被认为是DBSCAN的邻居,可选,'euclidean'	字符串	不必须	定数
metric_params	度量参数	默认为None,度量功能的其他关键字参数,可选字典	字典	不必须	定数
algorithm	算法	默认为auto,NearestNeighbors模块将使用该算法来计算逐点距离并查找最近的邻居。有关详细信息，请参见NearestNeighbors模块文档,可选'kd_tree','auto','brute','ball_tree'	字符串	不必须	定数
leaf_size	叶子大小	默认为30,叶大小传递给BallTree或cKDTree。这会影响构造和查询的速度，以及存储树所需的内存。最佳值取决于问题的性质,可选整数	整数	不必须	定数
p	p	默认为None,Minkowski度量标准的功效，用于计算点之间的距离,可选浮点数	浮点数	不必须	定数
n_jobs	CPU数量	默认为None,要运行的并行作业数。 'None'表示1，除非在：obj：'joblib.parallel_backend'上下文中。更多细节,可选整数	整数	不必须	定数

[output]
core_sample_indices_	核心样本指标	核心样本指标	一维数组
components_	组件	通过训练找到的每个核心样本的副本	二维数组
labels_	标签	数据集中给fit()的每个点的聚类标签。噪声样本的标签为-1	一维数组

[outline]
从向量数组或距离矩阵执行DBSCAN聚类。

[describe]
从向量数组或距离矩阵执行DBSCAN聚类。
DBSCAN-带有噪声的应用程序的基于密度的空间聚类。
查找高密度的核心样本并从中扩展聚类。
适用于包含相似密度簇的数据。

'''


def main(x_train, y_train,
         eps=0.5, min_samples=5, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=None
         ):
    if type(x_train) is str:
        x_train = eval(x_train)
    if type(y_train) is str:
        y_train = eval(y_train)
    if type(eps) is str:
        eps = eval(eps)
    if type(min_samples) is str:
        min_samples = eval(min_samples)
    if type(metric_params) is str:
        metric_params = eval(metric_params)
    if type(leaf_size) is str:
        leaf_size = eval(leaf_size)
    if type(p) is str:
        p = eval(p)
    if type(n_jobs) is str:
        n_jobs = eval(n_jobs)

    return r.run(x_train=x_train, y_train=y_train, eps=eps,
                 min_samples=min_samples,
                 metric=metric,
                 metric_params=metric_params,
                 algorithm=algorithm,
                 leaf_size=leaf_size,
                 p=p,
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