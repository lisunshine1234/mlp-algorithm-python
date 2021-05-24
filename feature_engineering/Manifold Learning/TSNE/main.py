import numpy as np
import run as  r
import json

'''
[id]
173

[name]
TSNE

[input]
array	数组	需要处理的数组	二维数组	必须	定数
label_index	标签列号	默认为None,表示所有列参与训练，输入整数代表标签所在的列号，可选整数	数字	不必须	定数
n_components	组件数	默认为2,嵌入式空间的尺寸,可选整数	整数	不必须	定数
perplexity	perplexity	默认为30.0,困惑与其他流形学习算法中使用的最近邻居的数量有关。较大的数据集通常需要较大的困惑度。考虑选择一个介于5到50之间的值。不同的值可能导致明显不同的结果,可选浮点数	浮点数	不必须	定数
early_exaggeration	early_exaggeration	默认为12.0,控制原始空间中的自然簇在嵌入式空间中的紧密程度以及它们之间有多少空间。对于较大的值，自然簇之间的空间在嵌入空间中会更大。同样，此参数的选择不是很关键。如果成本函数在初始优化过程中增加，则早期夸张因子或学习率可能会太高,可选浮点数	浮点数	不必须	定数
learning_rate	学习率	默认为200.0,t-SNE的学习率通常在[10.0，1000.0]范围内。如果学习率太高，则数据看起来可能像'ball'，并且任何点都与其最近的邻居大致等距。如果学习率太低，大多数点可能看起来像压缩在密集的云中，没有异常值。如果成本函数陷入不良的局部最小值中，则提高学习率可能会有所帮助,可选浮点数	浮点数	不必须	定数
n_iter	n_iter	默认为1000,优化的最大迭代次数。至少应为250,可选整数	整数	不必须	定数
n_iter_without_progress	n_iter_without_progress	默认为300,在中止优化之前没有进度的最大迭代次数，在250次具有早期夸张的初始迭代之后使用。请注意，仅每50次迭代检查一次进度，因此该值将舍入为50的下一个倍数,可选整数	整数	不必须	定数
min_grad_norm	min_grad_norm	默认为1e-7,如果梯度范数低于此阈值，则优化将停止,可选浮点数	浮点数	不必须	定数
metric	度量	默认为'euclidean',计算要素阵列中实例之间的距离时使用的度量。如果度量为'precomputed'，则假定X为距离矩阵。可调用对象应将X的两个数组作为输入，并返回一个指示它们之间距离的值。默认值为'euclidean'，它被解释为平方的欧几里德距离,可选字符串,字符串	字符串	不必须	定数
init	初始化方法	默认为'random',嵌入的初始化。可能的选项是'random'，'pca'和一个numpy形状的数组(n_samples，n_components)。 PCA初始化不能与预先计算的距离一起使用，并且通常比随机初始化更全局稳定,可选数组,,'random'	字符串	不必须	定数
verbose	详细程度	默认为0,详细程度,可选整数	整数	不必须	定数
random_state	随机种子	默认为None,确定随机数生成器。在多个函数调用之间传递int以获得可重复的结果。请注意，不同的初始化可能会导致不同的局部最低成本,可选整数	整数	不必须	定数
method	method	默认为barnes_hut,默认情况下，梯度计算算法使用以O(NlogN)时间运行的Barnes-Hut逼近。 method = 'exact'将在O(N ^ 2)时间内以较慢但精确的算法运行。当最近邻居误差需要大于3％时，应使用精确算法。但是，确切的方法无法扩展到数百万个示例,可选,'barnes_hut'	字符串	不必须	定数
angle	angle	默认为0.5,仅在method = 'barnes_hut'时使用。这是Barnes-Hut T-SNE在速度和精度之间的权衡。 'angle'是从点开始测量的远端节点的角度大小(在[3]中称为theta)。如果该大小小于'angle'，则将其用作其中包含的所有点的汇总节点。此方法对此参数在0.2-0.8范围内的变化不太敏感。小于0.2的角度会迅速增加计算时间，大于0.8的角度会迅速增加误差,可选浮点数	浮点数	不必须	定数
n_jobs	CPU数量	默认为None,为邻居搜索运行的并行作业数。当'metric=' precomputed ' or (' metric = 'euclidean'和'method=' exact '). ' None ' means 1 unless in a :obj:' joblib.parallel_backend'上下文时，此参数不起作用。更多细节,可选整数	整数	不必须	定数

[output]
array	数组	处理之后的数组	二维数组(数值)
embedding_	embedding_	存储嵌入向量	二维数组(数值)
kl_divergence_	kl_divergence_	优化后的Kullback-Leibler散度	浮点数
n_iter_	迭代次数	运行的迭代次数	整数

[outline]
t分布随机邻居嵌入。
t-SNE [1]是可视化高维数据的工具。
它将数据点之间的相似性转换为联合概率，并试图最小化低维嵌入和高维数据的联合概率之间的Kullback-Leibler差异。
t-SNE的成本函数不是凸的，即使用不同的初始化我们可以得到不同的结果。
如果要素数量非常多，强烈建议使用另一种降维方法(例如，对于密集数据使用PCA或对于稀疏数据使用TruncatedSVD)将尺寸数量减少到合理的数量(例如50个)。
这将抑制一些噪声并加快样本之间成对距离的计算。

[describe]
t分布随机邻居嵌入。
t-SNE [1]是可视化高维数据的工具。
它将数据点之间的相似性转换为联合概率，并试图最小化低维嵌入和高维数据的联合概率之间的Kullback-Leibler差异。
t-SNE的成本函数不是凸的，即使用不同的初始化我们可以得到不同的结果。
如果要素数量非常多，强烈建议使用另一种降维方法(例如，对于密集数据使用PCA或对于稀疏数据使用TruncatedSVD)将尺寸数量减少到合理的数量(例如50个)。
这将抑制一些噪声并加快样本之间成对距离的计算。

'''


def main(array, label_index=None, n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate=200.0, n_iter=1000, n_iter_without_progress=300,
         min_grad_norm=1e-7,
         metric="euclidean", init="random", verbose=0, random_state=None, method='barnes_hut', angle=0.5, n_jobs=None
         ):
    if type(array) is str:
        array = eval(array)
    if type(label_index) is str:
        label_index = eval(label_index)
    if type(n_components) is str:
        n_components = eval(n_components)
    if type(perplexity) is str:
        perplexity = eval(perplexity)
    if type(early_exaggeration) is str:
        early_exaggeration = eval(early_exaggeration)
    if type(learning_rate) is str:
        learning_rate = eval(learning_rate)
    if type(n_iter) is str:
        n_iter = eval(n_iter)
    if type(n_iter_without_progress) is str:
        n_iter_without_progress = eval(n_iter_without_progress)
    if type(min_grad_norm) is str:
        min_grad_norm = eval(min_grad_norm)
    if type(init) is str and init != 'random':
        init = eval(init)
    if type(verbose) is str:
        verbose = eval(verbose)
    if type(random_state) is str:
        random_state = eval(random_state)
    if type(angle) is str:
        angle = eval(angle)
    if type(n_jobs) is str:
        n_jobs = eval(n_jobs)
    return r.run(array=array,
                 label_index=label_index,
                 n_components=n_components,
                 perplexity=perplexity,
                 early_exaggeration=early_exaggeration,
                 learning_rate=learning_rate,
                 n_iter=n_iter,
                 n_iter_without_progress=n_iter_without_progress,
                 min_grad_norm=min_grad_norm,
                 metric=metric,
                 init=init,
                 verbose=verbose,
                 random_state=random_state,
                 method=method,
                 angle=angle,
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
