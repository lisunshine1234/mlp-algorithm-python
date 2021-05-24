import numpy as np
import run as  r

'''
[id]
142

[name]
AffinityPropagation

[input]
x_train	训练集	训练集标签数据集	二维数组	必须	定数
y_train	测试集	测试集数据集	二维数组	必须	定数
x_test	训练集标签	训练集标签标签	一维数组	必须	定数
damping	阻尼系数	默认为.5,阻尼系数(介于0.5和1之间)是相对于输入值(加权1-阻尼)保持当前值的程度。为了避免在更新这些值(消息)时出现数值振荡,可选浮点数	浮点数	不必须	定数
max_iter	最大迭代次数	默认为200,最大迭代次数,可选整数	整数	不必须	定数
convergence_iter	收敛迭代数	默认为15,停止收敛的估计簇数没有变化的迭代数,可选整数	整数	不必须	定数
copy	复制	默认为True,复制输入数据,可选布尔值	布尔值	不必须	定数
preference	偏好	默认为None,每个点的偏好-具有较大偏好值的点更有可能被选择为示例。样本数量(即集群数量)受输入首选项值的影响。如果首选项未作为参数传递，则将其设置为输入相似度的中位数,可选数组,浮点数	字符串	不必须	定数
affinity	亲和力	默认为euclidean,使用哪个亲和力。目前支持'precomputed'和'euclidean'。 'euclidean'使用点之间的负平方欧几里德距离,可选'euclidean','precomputed'	字符串	不必须	定数
verbose	详细程度	默认为False,是否冗长,可选布尔值	布尔值	不必须	定数
random_state	随机种子	默认为0,伪随机数发生器控制启动状态。将int用于在函数调用之间可重现的结果,可选整数	整数	不必须	定数

[output]
train_predict	预测	训练集预测结果	一维数组(数值)
test_predict	预测	测试集预测结果	一维数组(数值)
cluster_centers_indices_	cluster_centers_indices_	聚类中心指标	一维数组
cluster_centers_	聚类中心	聚类中心(如果affinity！= 'precomputed')	二维数组
labels_	labels_	每个点的标签	一维数组
affinity_matrix_	affinity_matrix_	存储在'fit'中使用的相似度矩阵	二维数组
n_iter_	迭代次数	收敛的迭代次数	整数

[outline]


[describe]
执行数据的相似性传播聚类。

'''


def main(x_train, y_train, x_test,
         damping=.5, max_iter=200, convergence_iter=15, copy=True,
         preference=None, affinity='euclidean', verbose=False, random_state=0
         ):
    if type(x_train) is str:
        x_train = eval(x_train)
    if type(y_train) is str:
        y_train = eval(y_train)
    if type(x_test) is str:
        x_test = eval(x_test)
    if type(damping) is str:
        damping = eval(damping)
    if type(max_iter) is str:
        max_iter = eval(max_iter)
    if type(convergence_iter) is str:
        convergence_iter = eval(convergence_iter)
    if type(copy) is str:
        copy = eval(copy)
    if type(preference) is str:
        preference = eval(preference)
    if type(verbose) is str:
        verbose = eval(verbose)
    if type(random_state) is str:
        random_state = eval(random_state)

    return r.run(x_train=x_train, y_train=y_train, x_test=x_test, damping=damping,
                 max_iter=max_iter,
                 convergence_iter=convergence_iter,
                 copy=copy,
                 preference=preference,
                 affinity=affinity,
                 verbose=verbose,
                 random_state=random_state)


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


