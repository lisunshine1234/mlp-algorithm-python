import numpy as np
import run as  r

'''
[id]
145

[name]
AgglomerativeClustering

[input]
x_train	训练集	训练集标签数据集	二维数组	必须	定数
y_train	测试集	测试集数据集	二维数组	必须	定数
n_clusters	簇数	默认为2,要查找的集群数。如果'None'不是'distance_threshold'，则必须为'None',可选整数	整数	不必须	定数
affinity	亲和力	默认为'euclidean',用于计算链接的度量。可以是'euclidean'，'l1'，'l2'，'manhattan'，'cosine'或'precomputed'。如果链接为'ward'，则仅接受'euclidean'。如果为'precomputed'，则需要​​距离矩阵(而不是相似度矩阵)作为拟合方法的输入,可选'euclidean'	字符串	不必须	定数
memory	memory	默认为None,用于缓存树计算的输出。默认情况下，不进行缓存。如果给出了字符串，则它是缓存目录的路径,可选整数,字符串	字符串	不必须	定数
connectivity	连通性	默认为None,连接矩阵。为每个样本定义遵循给定数据结构的相邻样本。这可以是连通性矩阵本身，也可以是将数据转换为连通性矩阵(例如从kneighbors_graph派生)的可调用对象。默认值为None，即分层聚类算法是非结构化的,可选数组	不定数组	不必须	定数
compute_full_tree	计算全树	默认为auto,尽早在n_clusters处停止树的构建。还要注意的是，当更改群集数量并使用缓存时，计算完整树可能是有利的。如果'True'不是'distance_threshold'，则必须为'None'。默认情况下，'compute_full_tree'是'auto'，当'True'不是'distance_threshold'或'None'次于100或'n_clusters'之间的最大值时，等于'0.02 * n_samples'。否则，'auto'等于'False',可选布尔值,'auto'	字符串	不必须	定数
linkage	链接标准	默认为ward,使用哪个链接标准。链接标准确定要在观察组之间使用的距离。该算法将合并最小化此标准的成对集群。-ward将合并的簇的方差最小化。-平均使用两组的每个观测值的距离的平均值。-完全或最大链接使用两组所有观测值之间的最大距离。-single使用两组的所有观测值之间的最小距离,可选'ward','average','single','complete'	字符串	不必须	定数
distance_threshold	距离阈值	默认为None,链接距离阈值，超过该距离时，群集将不会合并。如果不是'None'，则'n_clusters'必须为'None'，而'compute_full_tree'必须为'True',可选浮点数	浮点数	不必须	定数

[output]
n_clusters_	簇数	该算法找到的簇数。如果为'distance_threshold=None'，则等于给定的'n_clusters'	整数
labels_	标签	每个点的聚类标签	一维数组
n_leaves_	叶子数	层次树中的叶数	整数
n_connected_components_	组件连接数	图中估计的已连接组件数	整数
children_	children_	每个非叶节点的子级。小于'n_samples'的值对应于作为原始样本的树的叶子。大于或等于'i'的节点'n_samples'是非叶子节点，并且具有子节点'children_[i - n_samples]'。或者，在第i次迭代中，children [i] [0]和children [i] [1]合并以形成节点'n_samples + i	二维数组

[outline]
聚集聚类以递归方式合并这对最小增加给定链接距离的聚类对。

[describe]
聚集聚类以递归方式合并这对最小增加给定链接距离的聚类对。

'''


def main(x_train, y_train,
         n_clusters=2, affinity="euclidean", memory=None, connectivity=None, compute_full_tree='auto', linkage='ward', distance_threshold=None
         ):
    if type(x_train) is str:
        x_train = eval(x_train)
    if type(y_train) is str:
        y_train = eval(y_train)
    if type(n_clusters) is str:
        n_clusters = eval(n_clusters)
    if type(connectivity) is str:
        connectivity = eval(connectivity)
    if type(distance_threshold) is str:
        distance_threshold = eval(distance_threshold)
    return r.run(x_train=x_train, y_train=y_train,  n_clusters=n_clusters,
                 affinity=affinity,
                 memory=memory,
                 connectivity=connectivity,
                 compute_full_tree=compute_full_tree,
                 linkage=linkage,
                 distance_threshold=distance_threshold)


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