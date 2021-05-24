import numpy as np
import run as  r
import json

'''
[id]
170

[name]
LocallyLinearEmbedding

[input]
array	数组	需要处理的数组	二维数组	必须	定数
label_index	标签列号	默认为None,表示所有列参与训练，输入整数代表标签所在的列号，可选整数	数字	不必须	定数
n_neighbors	邻居数量	默认为5,每个点要考虑的邻居数量,可选整数,整数	字符串	不必须	定数
n_components	组件数	默认为2,流形的坐标数,可选整数,整数	字符串	不必须	定数
reg	reg	默认为1E-3,正则化常数，乘以距离的局部协方差矩阵的轨迹,可选浮点数	浮点数	不必须	定数
eigen_solver	eigen_solver	默认为auto,自动：算法将尝试为输入数据选择最佳方法。打包：在移位-反转模式下使用arnoldi迭代。对于此方法，M可以是稠密矩阵，稀疏矩阵或一般线性算子。警告：由于某些问题，ARPACK可能不稳定。最好尝试几个随机种子以检查结果。密集：使用标准密集矩阵运算进行特征值分解。对于此方法，M必须为数组或矩阵类型。对于大问题应避免使用此方法,可选,'dense','arpack','auto'	字符串	不必须	定数
tol	tol	默认为1E-6,'arpack'方法的公差如果eigen_solver == 'dense'，则不使用,可选浮点数	浮点数	不必须	定数
max_iter	最大迭代次数	默认为100,arpack求解器的最大迭代次数。如果eigen_solver == 'dense'不使用,可选整数,整数	字符串	不必须	定数
method	method	默认为standard,standard：使用标准的局部线性嵌入算法。参见参考文献[1]粗麻布：使用粗麻布特征图方法。此方法需要'n_neighbors > n_components * (1 + (n_components + 1) / 2'参见参考文献[2]修改：使用修改后的局部线性嵌入算法。请参见参考文献[3] ltsa：使用局部切线空间对齐算法请参见参考文献[4],可选,'standard','modified','ltsa','hessian'	字符串	不必须	定数
hessian_tol	hessian_tol	默认为1E-4,黑森州特征映射方法的公差。仅在'method == ' hessian'时使用,可选浮点数	浮点数	不必须	定数
modified_tol	modified_tol	默认为1E-12,修正的LLE方法的公差。仅在'method == ' modified'时使用,可选浮点数	浮点数	不必须	定数
neighbors_algorithm	neighbors_algorithm	默认为auto,用于最近邻居搜索的算法，传递给邻居。NearestNeighbors实例,可选,'brute','ball_tree','kd_tree','auto'	字符串	不必须	定数
random_state	随机种子	默认为None,当'eigen_solver' == 'arpack'时确定随机数生成器。传递整数以获得可重复的结果,可选整数	整数	不必须	定数
n_jobs	CPU数量	默认为None,要运行的并行作业数。 'None'表示1，除非在：obj：'joblib.parallel_backend'上下文中。更多细节,可选整数	整数	不必须	定数

[output]
array	数组	处理之后的数组	二维数组(数值)
embedding_	embedding_	存储嵌入向量	二维数组(数值)
reconstruction_error_	reconstruction_error_	与'embedding_'相关的重建错误	浮点数

[outline]
局部线性嵌入。

[describe]
局部线性嵌入。

'''


def main(array, label_index=None, n_neighbors=5, n_components=2, reg=1E-3, eigen_solver='auto', tol=1E-6, max_iter=100, method='standard', hessian_tol=1E-4,
         modified_tol=1E-12,
         neighbors_algorithm='auto', random_state=None, n_jobs=None
         ):
    if type(array) is str:
        array = eval(array)
    if type(label_index) is str:
        label_index = eval(label_index)
    if type(n_neighbors) is str:
        n_neighbors = eval(n_neighbors)
    if type(n_components) is str:
        n_components = eval(n_components)
    if type(reg) is str:
        reg = eval(reg)
    if type(tol) is str:
        tol = eval(tol)
    if type(max_iter) is str:
        max_iter = eval(max_iter)
    if type(hessian_tol) is str:
        hessian_tol = eval(hessian_tol)
    if type(modified_tol) is str:
        modified_tol = eval(modified_tol)
    if type(random_state) is str:
        random_state = eval(random_state)
    if type(n_jobs) is str:
        n_jobs = eval(n_jobs)
    return r.run(array=array,

                 label_index=label_index,
                 n_neighbors=n_neighbors,
                 n_components=n_components,
                 reg=reg,
                 eigen_solver=eigen_solver,
                 tol=tol,
                 max_iter=max_iter,
                 method=method,
                 hessian_tol=hessian_tol,
                 modified_tol=modified_tol,
                 neighbors_algorithm=neighbors_algorithm,
                 random_state=random_state,
                 n_jobs=n_jobs)


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
