import numpy as np
import run as  r

'''
[id]
132

[name]
KNeighborsRegressor

[input]
x_train	训练集	训练集标签数据集	二维数组	必须	定数
y_train	测试集	测试集数据集	二维数组	必须	定数
x_test	训练集标签	训练集标签标签	一维数组	必须	定数
y_test	测试集标签	测试集标签	一维数组	必须	定数
n_neighbors	查询的邻居数	默认为5,默认情况下用于：meth：'kneighbors'查询的邻居数,可选整数	整数	不必须	定数
weights	权重函数	默认为uniform,预测中使用的权重函数。可能的值：。-'uniform'：统一权重。每个邻域中的所有点均被加权。-'distance'：权重点按其距离的倒数表示。在这种情况下，查询点的近邻比远处的近邻具有更大的影响力。-[callable]：用户定义的函数，它接受距离数组，并返回包含权重的相同形状的数组。默认情况下使用统一权重,可选'distance','uniform'	字符串	不必须	定数
algorithm	算法	默认为auto,用于计算最近邻居的算法：。-'ball_tree'将使用：class：'BallTree'。-'kd_tree'将使用：class：'KDTree'。-'brute'将使用蛮力搜索。-'auto'将尝试根据传递给：meth：'fit'方法的值来决定最合适的算法。注意：在稀疏输入上进行拟合将使用蛮力覆盖此参数的设置,可选'ball_tree','auto','kd_tree','brute'	字符串	不必须	定数
leaf_size	叶子大小	默认为30,叶大小传递给BallTree或KDTree。这会影响构造和查询的速度，以及存储树所需的内存。最佳值取决于问题的性质,可选整数	整数	不必须	定数
p	p	默认为2,Minkowski指标的功率参数。当p = 1时，这等效于对p = 2使用manhattan_distance(l1)和euclidean_distance(l2)。对于任意p，使用minkowski_distance(l_p),可选整数	整数	不必须	定数
metric	metric	默认为minkowski,用于树的距离度量。默认度量标准为minkowski，p = 2等效于标准欧几里德度量标准。有关可用度量的列表，请参见：class：'DistanceMetric'的文档。如果度量为'precomputed'，则假定X为距离矩阵，并且在拟合过程中必须为正方形。 X可以是：term：'sparse graph'，在这种情况下，只有'nonzero'元素可以被视为邻居,可选'minkowski'	字符串	不必须	定数
metric_params	metric_params	默认为None,度量特征的其他关键字参数,可选字典	字典	不必须	定数
n_jobs	CPU数量	默认为None,为邻居搜索运行的并行作业数。 'None'表示1，除非在：obj：'joblib.parallel_backend'上下文中。更多细节。不适合方法,可选整数	整数	不必须	定数

[output]
train_predict	预测	训练集预测结果	一维数组(数值)
test_predict	预测	测试集预测结果	一维数组(数值)
train_score	正确率	训练集预测结果的正确率	数字
test_score	正确率	测试集预测结果的正确率	数字
effective_metric_	effective_metric_	使用的距离度量。它与'metric'参数或其同义词相同，例如'euclidean'如果'metric'参数设置为'minkowski'并且'p'参数设置为2	字符串
effective_metric_params_	effective_metric_params_	度量特征的其他关键字参数。对于大多数指标，'metric_params'参数将是相同的，但如果'p'属性设置为'effective_metric_'，则也可能包含'minkowski'参数值	字典

[outline]
基于k最近邻的回归。

[describe]
基于k最近邻的回归。
通过对训练集中最近邻居的目标进行局部插值来预测目标。

'''


def main(x_train, y_train, x_test, y_test,
         n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None,
         **kwargs
         ):
    if type(x_train) is str:
        x_train = eval(x_train)
    if type(y_train) is str:
        y_train = eval(y_train)
    if type(x_test) is str:
        x_test = eval(x_test)
    if type(y_test) is str:
        y_test = eval(y_test)
    if type(n_neighbors) is str:
        n_neighbors = eval(n_neighbors)
    if type(leaf_size) is str:
        leaf_size = eval(leaf_size)
    if type(p) is str:
        p = eval(p)
    if type(metric_params) is str:
        metric_params = eval(metric_params)
    if type(n_jobs) is str:
        n_jobs = eval(n_jobs)

    return r.run(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                 n_neighbors=n_neighbors,
                 weights=weights,
                 algorithm=algorithm,
                 leaf_size=leaf_size,
                 p=p,
                 metric=metric,
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
    back = main(x, y,x,y)

    print(back)
    for i in back:
        print(i + ":" + str(back[i]))

    json.dumps(back)
