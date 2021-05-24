import numpy as np
import run as  r

'''
[id]
147

[name]
OPTICS

[input]
x_train	训练集	训练集标签数据集	二维数组	必须	定数
y_train	测试集	测试集数据集	二维数组	必须	定数
x_test	训练集标签	训练集标签标签	一维数组	必须	定数
min_samples	最小样本	默认为5,一个点被视为核心点的邻域中的样本数。同样，在陡峭区域的上下可以't have more then'min_samples个连续的非陡峭点。表示为绝对数量或样本数量的分数(四舍五入至少为2),可选整数,浮点数	字符串	不必须	定数
max_eps	max_eps	默认为np.inf,一个样本的两个样本之间的最大距离应视为另一个样本的邻域。'np.inf'的默认值将标识所有规模的集群；减少'max_eps'将缩短运行时间,可选浮点数	浮点数	不必须	定数
metric	度量	默认为minkowski,用于距离计算的指标。可以使用scikit-learn或scipy.spatial.distance中的任何度量。如果metric是可调用的函数，则会在每对实例(行)上调用它，并记录结果值。可调用对象应将两个数组作为输入并返回一个值，指示它们之间的距离。指标的有效值为: ['cityblock','cosine','euclidean','l1','l2','manhattan'] ; ['braycurtis','canberra','chebyshev','correlation','dice','hamming','jaccard','kulsinski','mahalanobis','minkowski','rogerstanimoto','russellrao','seuclidean','sokalmier','yule']	字符串	不必须	定数
p	p	默认为2,Minkowski指标的参数来自：class：'sklearn.metrics.pairwise_distances'。当p = 1时，这等效于对p = 2使用manhattan_distance(l1)和euclidean_distance(l2)。对于任意p，使用minkowski_distance(l_p),可选整数	整数	不必须	定数
metric_params	度量参数	默认为None,度量功能的其他关键字参数,可选字典	字典	不必须	定数
cluster_method	聚类方法	默认为xi,用于使用计算的可达性和排序来提取群集的提取方法。可能的值为'xi'和'dbscan',可选'xi'	字符串	不必须	定数
eps	eps	默认为None,一个样本的两个样本之间的最大距离应视为另一个样本的邻域。默认情况下，它假定与'max_eps'相同的值。仅在'cluster_method='dbscan'时使用,可选浮点数	浮点数	不必须	定数
xi	最小陡度	默认为0.05,确定构成群集边界的可及性图上的最小陡度。例如，可达性图中的一个向上的点由从一个点到其后继的比率最多为1-xi的比率定义。仅在'cluster_method='xi'时使用,可选浮点数	浮点数	不必须	定数
predecessor_correction	前驱校正	默认为True,根据OPTICS [2] _计算出的前驱来校正聚类。此参数对大多数数据集影响最小。仅在'cluster_method='xi'时使用,可选布尔值	布尔值	不必须	定数
min_cluster_size	最小簇大小	默认为None,OPTICS群集中的最小样本数，表示为绝对数或样本数的分数(四舍五入至少为2)。如果为'None'，则使用'min_samples'的值。仅在'cluster_method='xi'时使用,可选整数,浮点数	字符串	不必须	定数
algorithm	算法	默认为auto,用于计算最近邻居的算法：。-'ball_tree'将使用：class：'BallTree'。-'kd_tree'将使用：class：'KDTree'。-'brute'将使用蛮力搜索。-'auto'将尝试根据传递给：meth：'fit'方法的值来决定最合适的算法。 (默认)注意：稀疏输入的拟合将使用蛮力覆盖此参数的设置,可选'brute','auto','kd_tree','ball_tree'	字符串	不必须	定数
leaf_size	叶子大小	默认为30,叶大小传递给：class：'BallTree'或：class：'KDTree'。这会影响构造和查询的速度，以及存储树所需的内存。最佳值取决于问题的性质,可选整数	整数	不必须	定数
n_jobs	CPU数量	默认为None,为邻居搜索运行的并行作业数。'None'表示1，除非在：obj：'joblib.parallel_backend'上下文中。更多细节,可选整数	整数	不必须	定数

[output]
labels_	标签	数据集中给fit()的每个点的聚类标签。未包含在'cluster_hierarchy_'的叶簇中的嘈杂样本和点被标记为-1	一维数组
reachability_	样本可达距离	每个样本的可达距离，按对象顺序索引。使用'clust.reachability_[clust.ordering_]'以群集顺序访问	一维数组
ordering_	排序列表	样本索引的集群排序列表	一维数组
core_distances_	核心点距离	每个样本成为核心点的距离，按对象顺序索引。永远不会成为核心的点之间的距离为inf。使用'clust.core_distances_[clust.ordering_]'以群集顺序访问	一维数组
predecessor_	到达样本的点	到达样本的点，按对象顺序索引。种子点的前身为-1	一维数组
cluster_hierarchy_	聚类层次结构	每行以'[start, end]'形式的集群列表，包括所有索引。群集按照'(end, -start)'(升序)排序，以便包含较小群集的较大群集紧随较小的群集之后。由于'labels_'不反映层次结构，因此通常为'len(cluster_hierarchy_) > np.unique(optics.labels_)'。还请注意，这些索引是'ordering_'的索引，即'X[ordering_][start:end + 1]'组成一个簇。仅在'cluster_method='xi'时可用	二维数组

[outline]
从向量数组估计聚类结构。

[describe]
从向量数组估计聚类结构。
与DBSCAN密切相关的OPTICS(用于识别聚类结构的订购点)可以找到高密度的核心样本并从中扩展聚类[1] _。
与DBSCAN不同，保留集群层次结构用于可变的邻域半径。
比当前的DBSCAN sklearn实现更适合用于大型数据集。
然后使用类DBSCAN方法(cluster_method ='dbscan')或[1] _(cluster_method ='xi')中提出的自动技术提取聚类。
此实现方式与原始OPTICS有所不同，它首先在所有点上执行k最近邻搜索以识别核心大小，然后在构造聚类顺序时仅计算到未处理点的距离。
注意，我们没有使用堆来管理扩展候选，因此时间复杂度将为O(n ^ 2)。

'''


def main(x_train, y_train,
         min_samples=5, max_eps=np.inf, metric='minkowski', p=2, metric_params=None, cluster_method='xi', eps=None, xi=0.05, predecessor_correction=True,
         min_cluster_size=None, algorithm='auto', leaf_size=30, n_jobs=None
         ):
    if type(x_train) is str:
        x_train = eval(x_train)
    if type(y_train) is str:
        y_train = eval(y_train)
    if type(min_samples) is str:
        min_samples = eval(min_samples)
    if type(max_eps) is str:
        max_eps = eval(max_eps)
    if type(p) is str:
        p = eval(p)
    if type(metric_params) is str:
        metric_params = eval(metric_params)
    if type(eps) is str:
        eps = eval(eps)
    if type(xi) is str:
        xi = eval(xi)
    if type(predecessor_correction) is str:
        predecessor_correction = eval(predecessor_correction)
    if type(min_cluster_size) is str:
        min_cluster_size = eval(min_cluster_size)
    if type(leaf_size) is str:
        leaf_size = eval(leaf_size)
    if type(n_jobs) is str:
        n_jobs = eval(n_jobs)

    return r.run(x_train=x_train, y_train=y_train, min_samples=min_samples,
                 max_eps=max_eps,
                 metric=metric,
                 p=p,
                 metric_params=metric_params,
                 cluster_method=cluster_method,
                 eps=eps,
                 xi=xi,
                 predecessor_correction=predecessor_correction,
                 min_cluster_size=min_cluster_size,
                 algorithm=algorithm,
                 leaf_size=leaf_size,
                 n_jobs=n_jobs
                 )


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