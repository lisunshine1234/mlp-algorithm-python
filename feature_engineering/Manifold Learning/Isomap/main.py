import numpy as np
import run as  r
import json

'''
[id]
169

[name]
Isomap

[input]
array	数组	需要处理的数组	二维数组	必须	定数
label_index	标签列号	默认为None,表示所有列参与训练，输入整数代表标签所在的列号，可选整数	数字	不必须	定数
n_neighbors	邻居数量	默认为5,每个点要考虑的邻居数量,可选整数,整数	字符串	不必须	定数
n_components	组件数	默认为2,流形的坐标数,可选整数,整数	字符串	不必须	定数
eigen_solver	eigen_solver	默认为auto,'auto'：尝试为给定问题选择最有效的求解器。 'arpack'：使用Arnoldi分解来找到特征值和特征向量。 'dense'：使用直接求解器(即LAPACK)进行特征值分解,可选'dense','arpack','auto'	字符串	不必须	定数
tol	tol	默认为0,收敛容差传递给了arpack或lobpcg。如果eigen_solver == 'dense'，则不使用,可选浮点数	浮点数	不必须	定数
max_iter	最大迭代次数	默认为None,arpack求解器的最大迭代次数。如果eigen_solver == 'dense'，则不使用,可选整数,整数	字符串	不必须	定数
path_method	path_method	默认为auto,查找最短路径的方法。 'auto'：尝试自动选择最佳算法。 'FW'：Floyd-Warshall算法。 'D'：Dijkstra的算法,可选,'D','auto','FW'	字符串	不必须	定数
neighbors_algorithm	neighbors_algorithm	默认为auto,用于最近邻居搜索的算法，传递给neighbors.NearestNeighbors实例,可选,'ball_tree','kd_tree','auto','brute'	字符串	不必须	定数
n_jobs	CPU数量	默认为None,要运行的并行作业数。 'None'表示1，可选整数	整数	不必须	定数
metric	度量	默认为minkowski,计算要素阵列中实例之间的距离时使用的度量。如果metric是字符串或可调用，则它必须是：func：'sklearn.metrics.pairwise_distances'为其metric参数所允许的选项之一。如果度量为'precomputed'，则假定X为距离矩阵，并且必须为正方形。 可选,'minkowski'	字符串	不必须	定数
p	p	默认为2,sklearn.metrics.pairwise.pairwise_distances中的Minkowski指标的参数。当p = 1时，这等效于对p = 2使用manhattan_distance(l1)和euclidean_distance(l2)。对于任意p，使用minkowski_distance(l_p),可选整数	整数	不必须	定数
metric_params	度量参数	默认为None,度量功能的其他关键字参数,可选字典	字典	不必须	定数

[output]
array	数组	处理之后的数组	二维数组(数值)
embedding_	embedding_	存储嵌入向量	二维数组(数值)
dist_matrix_	dist_matrix_	存储训练数据的测地距离矩阵	二维数组(数值)

[outline]
等轴测图嵌入等距映射的非线性降维。

[describe]
等轴测图嵌入等距映射的非线性降维。

'''


def main(array, label_index=None, n_neighbors=5, n_components=2, eigen_solver='auto', tol=0, max_iter=None, path_method='auto', neighbors_algorithm='auto', n_jobs=None,
         metric='minkowski', p=2, metric_params=None):
    if type(array) is str:
        array = eval(array)
    if type(label_index) is str:
        label_index = eval(label_index)
    if type(n_neighbors) is str:
        n_neighbors = eval(n_neighbors)
    if type(n_components) is str:
        n_components = eval(n_components)
    if type(tol) is str:
        tol = eval(tol)
    if type(max_iter) is str:
        max_iter = eval(max_iter)
    if type(n_jobs) is str:
        n_jobs = eval(n_jobs)
    if type(p) is str:
        p = eval(p)
    if type(metric_params) is str:
        metric_params = eval(metric_params)
    return r.run(array=array,
                 label_index=label_index,
                 n_neighbors=n_neighbors,
                 n_components=n_components,
                 eigen_solver=eigen_solver,
                 tol=tol,
                 max_iter=max_iter,
                 path_method=path_method,
                 neighbors_algorithm=neighbors_algorithm,
                 n_jobs=n_jobs,
                 metric=metric,
                 p=p,
                 metric_params=metric_params)


if __name__ == '__main__':
    import numpy as np
    import json

    array = np.loadtxt('D:\\123_2.csv', delimiter=',')
    array = array[0:20, :]

    y = array[:, -1].tolist()
    x = np.delete(array, -1, axis=1).tolist()
    array = array.tolist()
    back = main(array,label_index=-1)

    print(back)
    for i in back:
        print(i + ":" + str(back[i]))

    json.dumps(back)
