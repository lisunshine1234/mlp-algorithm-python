import numpy as np
import run as  r

'''
[id]
148

[name]
Birch

[input]
x_train	训练集	训练集标签数据集	二维数组	必须	定数
y_train	测试集	测试集数据集	二维数组	必须	定数
x_test	训练集标签	训练集标签标签	一维数组	必须	定数
threshold	阈值	默认为0.5,通过合并新样本和最接近的子集群而获得的子集群的半径应小于阈值。否则，将启动一个新的子集群。将此值设置得很低会促进分裂，反之亦然,可选浮点数	浮点数	不必须	定数
branching_factor	分支因子	默认为50,每个节点中CF子集群的最大数量。如果输入新样本，使得子集群的数量超过branching_factor，则该节点将被分为两个节点，每个子集群都将重新分配。删除了该节点的父子群集，并添加了两个新的子群集作为2个拆分节点的父节点,可选整数	整数	不必须	定数
n_clusters	簇数	默认为3,最后的聚类步骤之后的聚类数，该步骤将叶子中的子类视为新样本。-'None'：不执行最后的聚类步骤，并且按原样返回子集群。-：mod：'sklearn.cluster' Estimator：如果提供了模型，则该模型适合将子群集视为新样本，并将初始数据映射到最近的子群集的标签。-'int'：适合的模型是：'AgglomerativeClustering'，其中'n_clusters'设置为等于int,可选整数	整数	不必须	定数
compute_labels	计算标签	默认为True,是否为每个拟合计算标签,可选布尔值	布尔值	不必须	定数
copy	复制	默认为True,是否复制给定数据。如果设置为False，则初始数据将被覆盖,可选布尔值	布尔值	不必须	定数

[output]
train_predict	预测	训练集预测结果	一维数组(数值)
test_predict	预测	测试集预测结果	一维数组(数值)
subcluster_centers_	所有子类质心	直接从叶子读取的所有子类的质心	二维数组
subcluster_labels_	subcluster_labels_	全局分组后，分配给子群集的质心的标签	一维数组
labels_	labels_	分配给输入数据的标签数组。如果使用partial_fit代替fit，则将它们分配给最后一批数据	一维数组

[outline]
Birch聚类算法。

[describe]
实现Birch聚类算法。
它是一种内存有效的在线学习算法，可以替代MiniBatchKMeans类。
它构造一个树数据结构，其中簇质心从叶中读取。
这些可以是最终的聚类质心，也可以作为其他聚类算法(例如AgglomerativeClustering)的输入提供。

'''



def main(x_train, y_train, x_test,
         threshold=0.5, branching_factor=50, n_clusters=3, compute_labels=True, copy=True
         ):
    if type(x_train) is str:
        x_train = eval(x_train)
    if type(y_train) is str:
        y_train = eval(y_train)
    if type(x_test) is str:
        x_test = eval(x_test)
    if type(threshold) is str:
        threshold = eval(threshold)
    if type(branching_factor) is str:
        branching_factor = eval(branching_factor)
    if type(n_clusters) is str:
        n_clusters = eval(n_clusters)
    if type(compute_labels) is str:
        compute_labels = eval(compute_labels)
    if type(copy) is str:
        copy = eval(copy)

    return r.run(x_train=x_train, y_train=y_train, x_test=x_test, threshold = threshold,
    branching_factor = branching_factor,
    n_clusters = n_clusters,
    compute_labels = compute_labels,
    copy = copy
)


if __name__ == '__main__':
    import numpy as np
    import json

    array = np.loadtxt('D:\\123_2.csv', delimiter=',')
    array = array[0:20, :]

    y = array[:, -1].tolist()
    x = np.delete(array, -1, axis=1).tolist()
    array = array.tolist()
    back = main(x, y,x)

    print(back)
    for i in back:
        print(i + ":" + str(back[i]))

    json.dumps(back)

