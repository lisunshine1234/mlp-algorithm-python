import numpy as np
import run as  r

'''
[id]
00000

[name]
OPTICS

[input]
x	数据集	数据集	二维数组	必须	定数
y	标签	标签	一维数组	必须	定数
n_clusters	簇数	默认为3,棋盘格结构中行和列的簇数,可选整数	整数	不必须	定数
method	method	默认为bistochastic,将奇异向量归一化并将其转换为双簇的方法。可以是'scale'，'bistochastic'或'log'之一。作者建议使用'log'。但是，如果数据稀疏，则日志规范化将不起作用，这就是默认值为'bistochastic'的原因。 ..警告：：如果为'method=' log'，则数据必须稀疏,可选'log','scale','bistochastic'	字符串	不必须	定数
n_components	组件数	默认为6,要检查的奇异向量的数量,可选整数	整数	不必须	定数
n_best	n_best	默认为3,要将数据投影到其上的最佳奇异向量的数量,可选整数	整数	不必须	定数
svd_method	svd_method	默认为randomized,选择用于查找奇异矢量的算法。可能是'randomized'或'arpack'。如果使用'randomized'，可选'randomized','arpack'	字符串	不必须	定数
n_svd_vecs	n_svd_vecs	默认为None,用于计算SVD的向量数。随机分配`svd_method = arpack`时对应于`ncv`，而`svd_method`则对应于n_oversamples`,可选整数	整数	不必须	定数
mini_batch	mini_batch	默认为False,是否使用小批量k均值，速度更快，但可能会得到不同的结果,可选布尔值	布尔值	不必须	定数
init	初始化方法	默认为k-means++,一种k均值算法的初始化方法；默认为'k-means++',可选数组,数组,'k-means++','random'	字符串	不必须	定数
n_init	随机初始化数量	默认为10,用k-means算法尝试的随机初始化次数。如果使用小批量k均值，则选择最佳初始化，并且算法运行一次。否则，将为每次初始化和选择的最佳解决方案运行该算法,可选整数	整数	不必须	定数
random_state	随机种子	默认为None,用于随机化奇异值分解和k-means初始化。使用int可以确定随机性,可选整数	整数	不必须	定数

[output]
rows_	rows_	聚类的结果。如果群集'rows[i, r]'包含行'i'，则'r'为True。仅在调用'fit'后可用	二维数组(布尔)
columns_	columns_	聚类的结果，例如'rows'	二维数组(布尔)
row_labels_	row_labels_	行分区标签	一维数组(数值)
column_labels_	column_labels_	列分区标签	一维数组(数值)

[outline]
频谱双聚类

[describe]
频谱双聚类
在数据具有基础棋盘格结构的假设下对行和列进行分区。
例如，如果有两个行分区和三个列分区，则每行将属于三个bicluster，而每列将属于两个bicluster。
相应的行和列标签向量的外部乘积给出了这种棋盘结构。
'''


def main(x, y,
         n_clusters=3, method='bistochastic', n_components=6, n_best=3, svd_method='randomized', n_svd_vecs=None, mini_batch=False, init='k-means++', n_init=10,
         random_state=None):
    if type(x) is str:
        x = eval(x)
    if type(y) is str:
        y = eval(y)
    if type(n_clusters) is str:
        n_clusters = eval(n_clusters)
    if type(n_components) is str:
        n_components = eval(n_components)
    if type(n_best) is str:
        n_best = eval(n_best)
    if type(n_svd_vecs) is str:
        n_svd_vecs = eval(n_svd_vecs)
    if type(mini_batch) is str:
        mini_batch = eval(mini_batch)
    if type(init) is str and init != 'k-means++' and init != 'random':
        init = eval(init)
    if type(n_init) is str:
        n_init = eval(n_init)
    if type(random_state) is str:
        random_state = eval(random_state)

    return r.run(x=x, y=y,
                 n_clusters=n_clusters,
                 method=method,
                 n_components=n_components,
                 n_best=n_best,
                 svd_method=svd_method,
                 n_svd_vecs=n_svd_vecs,
                 mini_batch=mini_batch,
                 init=init,
                 n_init=n_init,
                 random_state=random_state)


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
    for i in back:
        print(i + ":" + str(back[i]))
    json.dumps(back)
