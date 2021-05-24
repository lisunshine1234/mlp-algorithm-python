import numpy as np
import run as  r

'''
[id]
107

[name]
RadiusNeighborsClassifier

[input]
x_train	训练集	训练集标签数据集	二维数组	必须	定数
y_train	测试集	测试集数据集	二维数组	必须	定数
x_test	训练集标签	训练集标签标签	一维数组	必须	定数
y_test	测试集标签	测试集标签	一维数组	必须	定数
radius	radius	默认为1.0,默认用于'radius_neighbors'查询的参数空间范围,可选浮点数	浮点数	不必须	定数
weights	权重函数	默认为uniform,预测中使用的权重函数。可能的值：。-'uniform'：统一权重。每个邻域中的所有点均被加权。-'distance'：权重点按其距离的倒数表示。在这种情况下，查询点的近邻比远处的近邻具有更大的影响力。-[callable]：一个用户定义的函数，它接受距离数组，并返回包含权重的相同形状的数组。默认情况下使用统一权重,可选'distance','uniform'	字符串	不必须	定数
algorithm	算法	默认为auto,用于计算最近邻居的算法：。-'ball_tree'将使用：class：'BallTree'。-'kd_tree'将使用：class：'KDTree'。-'brute'将使用蛮力搜索。-'auto'将尝试根据传递给：meth：'fit'方法的值来决定最合适的算法。注意：在稀疏输入上进行拟合将使用蛮力覆盖此参数的设置,可选'auto','brute','kd_tree','ball_tree'	字符串	不必须	定数
leaf_size	叶子大小	默认为30,叶大小传递给BallTree或KDTree。这会影响构造和查询的速度，以及存储树所需的内存。最佳值取决于问题的性质,可选整数	整数	不必须	定数
p	p	默认为2,Minkowski指标的功率参数。当p = 1时，这等效于对p = 2使用manhattan_distance(l1)和euclidean_distance(l2)。对于任意p，使用minkowski_distance(l_p),可选整数	整数	不必须	定数
metric	metric	默认为minkowski,用于树的距离度量。默认度量标准为minkowski，p = 2等效于标准欧几里德度量标准。有关可用度量的列表，请参见：class：'DistanceMetric'的文档。如果度量为'precomputed'，则假定X为距离矩阵，并且在拟合过程中必须为正方形。 X可以是：term：'sparse graph'，在这种情况下，只有'nonzero'元素可以被视为邻居,可选'minkowski'	字符串	不必须	定数
outlier_label	outlier_label	默认为None,离群样本的标签(给定半径中没有邻居的样本)。-手动标签：str或int标签(应与y类型相同)或使用多输出时的手动标签列表。-'most_frequent'：将y最频繁的标签分配给异常值。-None：当检测到任何异常值时，将引发ValueError,可选'most_frequent'	字符串	不必须	定数
metric_params	metric_params	默认为None,度量特征的其他关键字参数,可选字典	字典	不必须	定数
n_jobs	CPU数量	默认为None,为邻居搜索运行的并行作业数。 'None'表示1，除非在：obj：'joblib.parallel_backend'上下文中。更多细节,可选整数	整数	不必须	定数

[output]
train_predict	预测	训练集预测结果	一维数组(数值)
test_predict	预测	测试集预测结果	一维数组(数值)
train_score	正确率	训练集预测结果的正确率	数字
test_score	正确率	测试集预测结果的正确率	数字
classes_	类标签	分类器已知的类标签	一维数组
effective_metric_	effective_metric_	使用的距离度量。它与'metric'参数或其同义词相同，例如'euclidean'如果'metric'参数设置为'minkowski'并且'p'参数设置为2	字符串
effective_metric_params_	effective_metric_params_	度量特征的其他关键字参数。对于大多数指标，'metric_params'参数将是相同的，但如果'p'属性设置为'effective_metric_'，则也可能包含'minkowski'参数值	字典
outputs_2d_	outputs_2d_	在拟合期间'y' s形状为(n_samples，)或(n_samples，1)时为False，否则为True	布尔值

[outline]
分类器在给定半径内实现邻居之间的投票。

[describe]
分类器在给定半径内实现邻居之间的投票。

'''


def main(x_train, y_train, x_test, y_test,
         radius=1.0, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', outlier_label=None, metric_params=None, n_jobs=None
         , **kwargs):
    if type(x_train) is str:
        x_train = eval(x_train)
    if type(y_train) is str:
        y_train = eval(y_train)
    if type(x_test) is str:
        x_test = eval(x_test)
    if type(y_test) is str:
        y_test = eval(y_test)
    if type(radius) is str:
        radius = eval(radius)
    if type(leaf_size) is str:
        leaf_size = eval(leaf_size)
    if type(p) is str:
        p = eval(p)
    if type(outlier_label) is str:
        outlier_label = eval(outlier_label)
    if type(metric_params) is str:
        metric_params = eval(metric_params)
    if type(n_jobs) is str:
        n_jobs = eval(n_jobs)

    return r.run(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, radius=radius,
                 weights=weights,
                 algorithm=algorithm,
                 leaf_size=leaf_size,
                 p=p,
                 metric=metric,
                 outlier_label=outlier_label,
                 metric_params=metric_params,
                 n_jobs=n_jobs,
                 **kwargs)


if __name__ == '__main__':
    import numpy as np
    import json

    array = np.loadtxt('D:\\123_2.csv', delimiter=',')
    array = array[0:20, :]

    y = array[:, -1].tolist()
    x = np.delete(array, -1, axis=1).tolist()
    array = array.tolist()
    back = main(x, y, x, y)

    print(back)
    for i in back:
        print(i + ":" + str(back[i]))

    json.dumps(back)