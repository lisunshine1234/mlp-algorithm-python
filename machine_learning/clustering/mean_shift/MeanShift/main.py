import numpy as np
import run as  r

'''
[id]
143

[name]
MeanShift

[input]
x_train	训练集	训练集标签数据集	二维数组	必须	定数
y_train	测试集	测试集数据集	二维数组	必须	定数
x_test	训练集标签	训练集标签标签	一维数组	必须	定数
bandwidth	内核带宽	默认为None,RBF内核中使用的带宽。如果未给出，则使用sklearn.cluster.estimate_bandwidth;估计带宽,可选浮点数	浮点数	不必须	定数
seeds	初始化内核种子	默认为None,用于初始化内核的种子。如果未设置，则通过clustering.get_bin_seeds计算种子，并使用带宽作为网格大小以及其他参数的默认值,可选数组	二维数组	不必须	定数
bin_seeding	初始内核位置	默认为False,如果为true，则初始内核位置不是所有点的位置，而是点的离散版本的位置，在此点上，点被合并到其粗糙度与带宽相对应的网格中。将此选项设置为True可以加快算法的速度，因为初始化的种子较少。默认值为False。如果Seeds参数不为None则忽略,可选布尔值	布尔值	不必须	定数
min_bin_freq	min_bin_freq	默认为1,为了加速算法，仅接受那些至少具有min_bin_freq点的容器作为种子,可选整数	整数	不必须	定数
cluster_all	聚类全部	默认为True,如果为true，则所有点都将聚类，即使那些不在任何内核中的孤儿也是如此。孤儿被分配到最近的内核。如果为false，则为孤儿分配群集标签-1,可选布尔值	布尔值	不必须	定数
n_jobs	CPU数量	默认为None,用于计算的作业数。通过计算并行运行的每个n_init来工作。 'None'表示1，除非在：obj：'joblib.parallel_backend'上下文中。更多细节,可选整数	整数	不必须	定数
max_iter	最大迭代次数	默认为300,如果尚未收敛，则在群集操作终止之前(针对该种子点)每个种子点的最大迭代次数,可选整数	整数	不必须	定数

[output]
train_predict	预测	训练集预测结果	一维数组(数值)
test_predict	预测	测试集预测结果	一维数组(数值)
cluster_centers_	聚类中心	集群中心的坐标	二维数组
labels_	labels_	每个点的标签	一维数组
n_iter_	迭代次数	对每个种子执行的最大迭代次数	整数

[outline]
使用平内核的均值漂移聚类。

[describe]
使用平内核的均值漂移聚类。
均值漂移聚类旨在发现平滑密度的样本中的“斑点”。
这是基于质心的算法，它通过将质心的候选更新为给定区域内点的均值来工作。
然后在后处理阶段对这些候选对象进行过滤，以消除几乎重复的部分，从而形成最终的形心集。
使用分级技术来实现可扩展性的播种。

'''


def main(x_train, y_train, x_test,
         bandwidth=None, seeds=None, bin_seeding=False, min_bin_freq=1, cluster_all=True, n_jobs=None, max_iter=300
         ):
    if type(x_train) is str:
        x_train = eval(x_train)
    if type(y_train) is str:
        y_train = eval(y_train)
    if type(x_test) is str:
        x_test = eval(x_test)
    if type(bandwidth) is str:
        bandwidth = eval(bandwidth)
    if type(seeds) is str:
        seeds = eval(seeds)
    if type(bin_seeding) is str:
        bin_seeding = eval(bin_seeding)
    if type(min_bin_freq) is str:
        min_bin_freq = eval(min_bin_freq)
    if type(cluster_all) is str:
        cluster_all = eval(cluster_all)
    if type(n_jobs) is str:
        n_jobs = eval(n_jobs)
    if type(max_iter) is str:
        max_iter = eval(max_iter)

    return r.run(x_train=x_train, y_train=y_train, x_test=x_test, bandwidth=bandwidth,
                 seeds=seeds,
                 bin_seeding=bin_seeding,
                 min_bin_freq=min_bin_freq,
                 cluster_all=cluster_all,
                 n_jobs=n_jobs,
                 max_iter=max_iter)

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
