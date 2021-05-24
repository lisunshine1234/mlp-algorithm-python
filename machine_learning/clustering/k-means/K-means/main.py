import numpy as np
import run as  r

'''
[id]
140

[name]
KMeans

[input]
x_train	训练集	训练集标签数据集	二维数组	必须	定数
y_train	测试集	测试集数据集	二维数组	必须	定数
x_test	训练集标签	训练集标签标签	一维数组	必须	定数
y_test	测试集标签	测试集标签	一维数组	必须	定数
n_clusters	簇数	默认为8,形成的簇数以及生成的质心数,可选整数	整数	不必须	定数
init	初始化方法	默认为k-means++,初始化方法：'k-means++'：以一种聪明的方式为k均值聚类选择初始聚类中心，以加快收敛速度​​。'random'：从初始质心的数据中随机选择'n_clusters'个观测值(行)。则其形状应为(n_clusters，n_features)并给出初始中心。如果传递了callable，则应使用参数X，n_clusters和随机状态并返回初始化,可选数组,数组,'k-means++','random'	字符串	不必须	定数
n_init	运行次数	默认为10,k均值算法将在不同质心种子下运行的次数。就惯性而言，最终结果将是n_init个连续运行的最佳输出,可选整数	整数	不必须	定数
max_iter	最大迭代次数	默认为300,单次运行的k均值算法的最大迭代次数,可选整数	整数	不必须	定数
tol	tol	默认为1e-4,关于Frobenius范数的相对公差，该范数表示两个连续迭代的聚类中心的差异，以声明收敛,可选浮点数	浮点数	不必须	定数
verbose	详细程度	默认为0,详细模式,可选整数	整数	不必须	定数
random_state	随机种子	默认为None,确定质心初始化的随机数生成。使用int可以确定随机性,可选整数	整数	不必须	定数
copy_x	copy_x	默认为True,当预先计算距离时，从数值上更精确地确定数据居中。如果copy_x为True(默认值)，则原始数据不会被修改。如果为False，则会修改原始数据，并在函数返回之前放回原始数据，但是可以通过减去然后加上数据均值来引入小的数值差异,可选布尔值	布尔值	不必须	定数
algorithm	算法	默认为auto,使用K均值算法。经典的EM风格算法是'full'。通过使用三角形不等式，'elkan'变化对于具有定义良好的聚类的数据更有效。但是，它会自动更改，但为了更好的启发式功能，将来可能会更改,可选'elkan','auto','full'	字符串	不必须	定数

[output]
train_predict	预测	训练集预测结果	一维数组(数值)
test_predict	预测	测试集预测结果	一维数组(数值)
train_score	正确率	训练集预测结果的正确率	数字
test_score	正确率	测试集预测结果的正确率	数字
cluster_centers_	聚类中心	聚类中心的坐标。如果算法在完全收敛之前停止运行(请参阅'tol'和'max_iter')，则这些将与'labels_'不一致	二维数组
labels_	labels_	每个点的标签	一维数组
inertia_	平方距离的总和	样本到其最近的聚类中心的平方距离的总和	浮点数
n_iter_	迭代次数	运行的迭代次数	整数

[outline]
K-均值聚类。

[describe]
K-均值聚类。

'''


def main(x_train, y_train, x_test, y_test,
         n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=1e-4,
         verbose=0, random_state=None, copy_x=True,
         algorithm='auto'
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
    if type(n_init) is str:
        n_init = eval(n_init)
    if type(max_iter) is str:
        max_iter = eval(max_iter)
    if type(tol) is str:
        tol = eval(tol)
    if type(verbose) is str:
        verbose = eval(verbose)
    if type(random_state) is str:
        random_state = eval(random_state)
    if type(copy_x) is str:
        copy_x = eval(copy_x)

    return r.run(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, n_clusters=n_clusters,
                 init=init,
                 n_init=n_init,
                 max_iter=max_iter,
                 tol=tol,
                 verbose=verbose,
                 random_state=random_state,
                 copy_x=copy_x,
                 algorithm=algorithm)


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
