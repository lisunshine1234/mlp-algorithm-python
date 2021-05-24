import numpy as np
import run as  r
import json

'''
[id]
172

[name]
SpectralEmbedding

[input]
array	数组	需要处理的数组	二维数组	必须	定数
label_index	标签列号	默认为None,表示所有列参与训练，输入整数代表标签所在的列号，可选整数	数字	不必须	定数
n_components	组件数	默认为2,投影子空间的尺寸,可选整数,整数	字符串	不必须	定数
affinity	亲和力	默认为'nearest_neighbors',如何构造亲和力矩阵。 。-'nearest_neighbors'：通过计算最近邻居的图来构造亲和矩阵。 。-'rbf'：通过计算径向基函数(RBF)内核构造亲和矩阵。 。-'precomputed'：将'X'解释为预先计算的亲和力矩阵。 。-'precomputed_nearest_neighbors'：将'X'解释为预先计算的最近邻居的稀疏图，并通过选择'n_neighbors'最近邻居构建亲和力矩阵。可调用：将传入的函数用作亲和力，该函数接收数据矩阵(n_samples，n_features)并返回亲和力矩阵(n_samples，n_samples),可选,'nearest_neighbors'	字符串	不必须	定数
gamma	gamma	默认为None,rbf内核的内核系数,可选浮点数	浮点数	不必须	定数
random_state	随机种子	默认为None,确定当'solver' == 'amg'时用于lobpcg特征向量初始化的随机数生成器。在多个函数调用之间传递int以获得可重复的结果,可选整数	整数	不必须	定数
eigen_solver	eigen_solver	默认为None,使用特征值分解策略。 AMG需要安装pyamg。在非常大的，稀疏的问题上它可以更快,可选'amg','arpack','lobpcg'	字符串	不必须	定数
n_neighbors	邻居数量	默认为None,最近邻图构建的最近邻数,可选整数	整数	不必须	定数
n_jobs	CPU数量	默认为None,要运行的并行作业数。 'None'表示1，除非在：obj：'joblib.parallel_backend'上下文中。更多细节,可选整数	整数	不必须	定数

[output]
array	数组	处理之后的数组	二维数组(数值)
embedding_	embedding_	训练矩阵的频谱嵌入	二维数组(数值)
affinity_matrix_	affinity_matrix_	Affinity_matrix由样本构成或预先计算	二维数组(数值)
n_neighbors_	n_neighbors_	有效使用的最近邻居的数量	整数

[outline]
频谱嵌入用于非线性降维。
形成由指定函数给定的亲和矩阵，并将光谱分解应用于相应的图拉普拉斯图。
对于每个数据点，通过特征向量的值给出转换结果。

[describe]
频谱嵌入用于非线性降维。
形成由指定函数给定的亲和矩阵，并将光谱分解应用于相应的图拉普拉斯图。
对于每个数据点，通过特征向量的值给出转换结果。

'''


def main(array, label_index=None, n_components=2, affinity="nearest_neighbors", gamma=None, random_state=None, eigen_solver=None, n_neighbors=None, n_jobs=None
         ):
    if type(array) is str:
        array = eval(array)
    if type(label_index) is str:
        label_index = eval(label_index)
    if type(n_components) is str:
        n_components = eval(n_components)
    if type(gamma) is str:
        gamma = eval(gamma)
    if type(random_state) is str:
        random_state = eval(random_state)
    if type(n_jobs) is str:
        n_jobs = eval(n_jobs)
    return r.run(array=array,
                 label_index=label_index,
                 n_components=n_components,
                 affinity=affinity,
                 gamma=gamma,
                 random_state=random_state,
                 eigen_solver=eigen_solver,
                 n_neighbors=n_neighbors,
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
