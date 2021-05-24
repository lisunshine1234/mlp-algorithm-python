import numpy as np
import run as  r
import json

'''
[id]
171

[name]
MDS

[input]
array	数组	需要处理的数组	二维数组	必须	定数
label_index	标签列号	默认为None,表示所有列参与训练，输入整数代表标签所在的列号，可选整数	数字	不必须	定数
n_components	组件数	默认为2,沉浸差异的维数,可选整数	整数	不必须	定数
metric	度量	默认为True,如果为'True'，则执行公制MDS；否则，执行非度量MDS,可选布尔值	布尔值	不必须	定数
n_init	随机初始化数量	默认为4,SMACOF算法将使用不同的初始化运行的次数。最终结果将是运行的最佳输出，取决于最终应力最小的运行,可选整数	整数	不必须	定数
max_iter	最大迭代次数	默认为300,单次运行的SMACOF算法的最大迭代次数,可选整数	整数	不必须	定数
verbose	详细程度	默认为0,详细程度,可选整数	整数	不必须	定数
eps	eps	默认为1e-3,关于应力的相对公差，在该应力下可以收敛,可选浮点数	浮点数	不必须	定数
n_jobs	CPU数量	默认为None,用于计算的作业数。如果使用多个初始化('n_init')，则算法的每次运行都是并行计算的。 'None'表示1，除非在：obj：'joblib.parallel_backend'上下文中。更多细节,可选整数	整数	不必须	定数
random_state	随机种子	默认为None,确定用于初始化中心的随机数生成器。在多个函数调用之间传递int以获得可重复的结果,可选整数	整数	不必须	定数
dissimilarity	dissimilarity	默认为'euclidean',要使用的相异性度量：。-'euclidean'：数据集中点之间的成对欧几里得距离。 。-'precomputed'：将预先计算的差异直接传递给'fit'和'fit_transform',可选'precomputed','euclidean'	字符串	不必须	定数

[output]
array	数组	处理之后的数组	二维数组(数值)
embedding_	embedding_	将数据集的位置存储在嵌入空间中	二维数组(数值)
stress_	stress_	应力的最终值(视差的平方距离与所有约束点的距离的平方和)	浮点数

[outline]
多维缩放。

[describe]
多维缩放。

'''


def main(array, label_index=None, n_components=2, metric=True, n_init=4, max_iter=300, verbose=0, eps=1e-3, n_jobs=None, random_state=None,
         dissimilarity="euclidean"
         ):
    if type(array) is str:
        array = eval(array)
    if type(label_index) is str:
        label_index = eval(label_index)
    if type(n_components) is str:
        n_components = eval(n_components)
    if type(metric) is str:
        metric = eval(metric)
    if type(n_init) is str:
        n_init = eval(n_init)
    if type(max_iter) is str:
        max_iter = eval(max_iter)
    if type(verbose) is str:
        verbose = eval(verbose)
    if type(eps) is str:
        eps = eval(eps)
    if type(n_jobs) is str:
        n_jobs = eval(n_jobs)
    if type(random_state) is str:
        random_state = eval(random_state)
    return r.run(array=array,

                 label_index=label_index,
                 n_components=n_components,
                 metric=metric,
                 n_init=n_init,
                 max_iter=max_iter,
                 verbose=verbose,
                 eps=eps,
                 n_jobs=n_jobs,
                 random_state=random_state,
                 dissimilarity=dissimilarity)


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
