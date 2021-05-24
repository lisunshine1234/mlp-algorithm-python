import numpy as np
import run as  r

'''
[id]
141

[name]
MiniBatchKMeans

[input]
x_train	训练集	训练集标签数据集	二维数组	必须	定数
y_train	测试集	测试集数据集	二维数组	必须	定数
x_test	训练集标签	训练集标签标签	一维数组	必须	定数
y_test	测试集标签	测试集标签	一维数组	必须	定数
n_clusters	簇数	默认为8,形成的簇数以及生成的质心数,可选整数	整数	不必须	定数
init	初始化方法	默认为k-means++,初始化方法'k-means++'：以智能方式选择初始聚类中心以进行k均值聚类，以加快收敛速度​​。有关更多详细信息，请参见k_init中的注释部分。 'random'：从初始质心的数据中随机选择k个观测值(行)。如果通过ndarray，则其形状应为(n_clusters，n_features)并给出初始中心,可选数组,数组,'random','k-means++'	字符串	不必须	定数
max_iter	最大迭代次数	默认为100,独立于任何早期停止准则试探法而停止之前，整个数据集上的最大迭代次数,可选整数	整数	不必须	定数
batch_size	批次大小	默认为100,迷你批次的大小,可选整数	整数	不必须	定数
verbose	详细程度	默认为0,详细模式,可选整数	整数	不必须	定数
compute_labels	计算标签	默认为True,一旦小批量优化收敛到合适状态，就可以为整个数据集计算标签分配和惯性,可选布尔值	布尔值	不必须	定数
random_state	随机种子	默认为None,确定质心初始化和随机重新分配的随机数生成。使用int可以确定随机性,可选整数	整数	不必须	定数
tol	tol	默认为0.0,根据相对中心变化来控制提前停止，该相对中心变化是通过对均方中心位置变化进行平滑，方差归一化而测得的。这种提早停止的启发式算法更接近于算法的批处理变体中使用的启发式算法，但是在惯性启发式算法上会产生少量的计算和内存开销。要基于归一化的中心更改禁用收敛检测，请将​​Tol设置为0.0(默认值),可选浮点数	浮点数	不必须	定数
max_no_improvement	最大没有改善	默认为10,基于连续的小批量数量控制提前停止，这不会使平滑惯量有所改善。要禁用基于惯性的收敛检测，请将​​max_no_improvement设置为None,可选整数	整数	不必须	定数
init_size	初始大小	默认为None,为了加快初始化速度而随机采样的样本数(有时会牺牲准确性)：唯一的算法是通过在数据的随机子集上运行批处理KMeans来初始化的。这需要大于n_clusters。如果'None'，则'init_size= 3 * batch_size',可选整数	整数	不必须	定数
n_init	随机初始化数量	默认为3,尝试的随机初始化的数量。与KMeans相比，该算法仅运行一次，使用惯性测量的'n_init'最佳初始化,可选整数	整数	不必须	定数
reassignment_ratio	最大计数数量的分数	默认为0.01,控制要重新分配的中心的最大计数数量的分数。较高的值意味着更容易重新分配计数较低的中心，这意味着模型将花费更长的时间进行收敛，但是应该收敛于更好的聚类,可选浮点数	浮点数	不必须	定数

[output]
train_predict	预测	训练集预测结果	一维数组(数值)
test_predict	预测	测试集预测结果	一维数组(数值)
train_score	正确率	训练集预测结果的正确率	数字
test_score	正确率	测试集预测结果的正确率	数字
cluster_centers_	聚类中心	集群中心的坐标	二维数组
labels_	labels_	每个点的标签(如果将compute_labels设置为True)	整数
inertia_	平方距离的总和	与所选分区相关联的惯性标准的值(如果compute_labels设置为True)。惯性定义为样本到其最近邻居的平方距离之和	浮点数

[outline]


[describe]
迷你批次K均值聚类。

'''


def main(x_train, y_train, x_test, y_test,
         n_clusters=8, init='k-means++', max_iter=100, batch_size=100, verbose=0, compute_labels=True, random_state=None, tol=0.0, max_no_improvement=10,
         init_size=None, n_init=3, reassignment_ratio=0.01
         ):
    if type(x_train) is str:
        x_train = eval(x_train)
    if type(y_train) is str:
        y_train = eval(y_train)
    if type(x_test) is str:
        x_test = eval(x_test)
    if type(y_test) is str:
        y_test = eval(y_test)
    if type(n_clusters) is str:
        n_clusters = eval(n_clusters)
    if type(init) is str and init != 'k-means++' and init != 'random':
        init = eval(init)
    if type(max_iter) is str:
        max_iter = eval(max_iter)
    if type(batch_size) is str:
        batch_size = eval(batch_size)
    if type(verbose) is str:
        verbose = eval(verbose)
    if type(compute_labels) is str:
        compute_labels = eval(compute_labels)
    if type(random_state) is str:
        random_state = eval(random_state)
    if type(tol) is str:
        tol = eval(tol)
    if type(max_no_improvement) is str:
        max_no_improvement = eval(max_no_improvement)
    if type(init_size) is str:
        init_size = eval(init_size)
    if type(n_init) is str:
        n_init = eval(n_init)
    if type(reassignment_ratio) is str:
        reassignment_ratio = eval(reassignment_ratio)
    return r.run(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, n_clusters=n_clusters,
                 init=init,
                 max_iter=max_iter,
                 batch_size=batch_size,
                 verbose=verbose,
                 compute_labels=compute_labels,
                 random_state=random_state,
                 tol=tol,
                 max_no_improvement=max_no_improvement,
                 init_size=init_size,
                 n_init=n_init,
                 reassignment_ratio=reassignment_ratio
                 )


if __name__ == '__main__':
    import numpy as np
    import json

    array = np.loadtxt('D:\\123_2.csv', delimiter=',')
    array = array[0:20, :]

    y = array[:, -1].tolist()
    x = np.delete(array, -1, axis=1).tolist()
    array = array.tolist()
    back = main(x, y,x,y)

    print(back)
    for i in back:
        print(i + ":" + str(back[i]))

    json.dumps(back)