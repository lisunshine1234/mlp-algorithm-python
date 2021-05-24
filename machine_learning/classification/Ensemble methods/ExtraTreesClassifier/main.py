import numpy as np
import run as  r

'''
[id]
158

[name]
ExtraTreesClassifier

[input]
x_train	训练集	训练集标签数据集	二维数组	必须	定数
y_train	测试集	测试集数据集	二维数组	必须	定数
x_test	训练集标签	训练集标签标签	一维数组	必须	定数
y_test	测试集标签	测试集标签	一维数组	必须	定数
n_estimators	n_estimators	默认为100,森林中树木的数量,可选整数	整数	不必须	定数
criterion	标准	默认为'gini',衡量分割质量的功能。支持的标准是：基尼杂质'gini'和信息增益'entropy',可选'gini','entropy'	字符串	不必须	定数
max_depth	最大深度	默认为None,树的最大深度。如果为None，则将节点展开，直到所有叶子都是纯净的，或者直到所有叶子都包含少于min_samples_split个样本,可选整数	整数	不必须	定数
min_samples_split	拆分最小样本数	默认为2,拆分内部节点所需的最小样本数：。-如果为int，则将'min_samples_split'作为最小数。-如果是浮点型，则'min_samples_split'是分数，而'ceil(min_samples_split * n_samples)'是每个拆分的最小样本数,可选整数,浮点数	字符串	不必须	定数
min_samples_leaf	最小样本数	默认为1,在叶节点处所需的最小样本数。仅在任何深度的分裂点在左分支和右分支中的每个分支上至少留下'min_samples_leaf'个训练样本时，才考虑。这可能具有平滑模型的效果，尤其是在回归中。-如果为int，则将'min_samples_leaf'作为最小数字。-如果为float，则'min_samples_leaf'是分数，而'ceil(min_samples_leaf * n_samples)'是每个节点的最小样本数,可选整数,浮点数	字符串	不必须	定数
min_weight_fraction_leaf	最小重量分数叶子	默认为0.,在所有叶节点处(所有输入样本)的权重总和中的最小加权分数。如果未提供sample_weight，则样本的权重相等,可选浮点数	浮点数	不必须	定数
max_features	最佳分割特征数量	默认为'auto',寻找最佳分割时要考虑的要素数量：。-如果为int，则在每个分割中考虑'max_features'个要素。-如果为float，则'max_features'为小数，并且在每个拆分中均考虑'int(max_features * n_features)'特征。-如果是'auto'，则为'max_features=sqrt(n_features)'。-如果为'sqrt'，则为'max_features=sqrt(n_features)'。-如果是'log2'，则为'max_features=log2(n_features)'。-如果为None，则'max_features=n_features'。注意：在找到至少一个有效的节点样本分区之前，分割的搜索不会停止，即使它需要有效地检查'max_features'个以上的特征,可选整数,浮点数,'sqrt','auto','log2'	字符串	不必须	定数
max_leaf_nodes	最大叶节点	默认为None,以'max_leaf_nodes'优先的方式种植$。最佳节点定义为杂质的相对减少。如果为None，则叶节点数不受限制,可选整数	整数	不必须	定数
min_impurity_decrease	最小杂质减少	默认为0.,如果节点分裂会导致杂质的减少大于或等于该值，则该节点将被分裂。加权杂质减少方程如下：N_t / N *(杂质-N_t_R / N_t * right_impurity。-N_t_L / N_t * left_impurity)其中'N'是样本总数，'N_t'是当前节点的样本数， 'N_t_L'是左子项中的样本数，'N_t_R'是右子项中的样本数。如果传递了'N'，则'N_t'，'N_t_R'，'N_t_L'和'sample_weight'均指加权总和,可选浮点数	浮点数	不必须	定数
min_impurity_split	最小杂质分裂	默认为None,树木生长尽早停止的阈值。如果节点的杂质高于阈值，则该节点将分裂，否则为叶,可选浮点数	浮点数	不必须	定数
bootstrap	bootstrap	默认为False,建立树木时是否使用引导程序样本。如果为False，则将整个数据集用于构建每棵树,可选布尔值,字符串	字符串	不必须	定数
oob_score	oob_score	默认为False,是否使用现成的样本来估计泛化精度,可选布尔值	布尔值	不必须	定数
n_jobs	CPU数量	默认为None,要并行运行的作业数。 ：meth：'fit' 、: meth：'predict' 、: meth：'decision_path'和：meth：'apply'都在树上并行化。 'None'表示1，除非在：obj：'joblib.parallel_backend' <n_jobs>'中获取更多详细信息,可选整数	整数	不必须	定数
random_state	随机种子	默认为None,控制3个随机性源：。-构建树时使用的示例的自举(如果'bootstrap=True')。-在每个节点上寻找最佳分割时要考虑的特征采样(如果'max_features < n_features')。-分割的绘制对于每个'max_features,可选整数	整数	不必须	定数
verbose	详细程度	默认为0,在拟合和预测时控制详细程度,可选整数	整数	不必须	定数
warm_start	warm_start	默认为False,设置为'True'时，请重用上一个调用的解决方案以拟合并在集合中添加更多估计量，否则，仅拟合一个整体,可选布尔值	布尔值	不必须	定数
class_weight	类别权重	默认为None,与类关联的权重，格式为'{class_label: weight}'。如果未给出，则所有类均应具有权重一。对于多输出问题，可以按与y列相同的顺序提供字典列表。请注意，对于多输出(包括多标签)，应在其自己的字典中为每列的每个类定义权重。例如，对于四类多标签分类，权重应为[{0：1、1：1：1}，{0：1、1：5}，{0：1、1：1：1}，{0：1、1： 1}]，而不是[{1：1}，{2：5}，{3：1}，{4：1}]。 'balanced'模式使用y的值自动将与输入数据中的类频率成反比的权重调整为'n_samples / (n_classes * np.bincount(y))'。'balanced_subsample'模式与'balanced'相同，不同之处在于，权重是根据每个生长的树的引导程序样本计算的。对于多输出，y的每一列的权重将相乘。请注意，如果指定了sample_weight，则这些权重将与sample_weight(通过fit方法传递)相乘,可选list,字典,'balanced','balanced_subsample'	字符串	不必须	定数
ccp_alpha	复杂性参数	默认为0.0,用于最小化成本复杂性修剪的复杂性参数。将选择成本复杂度最大且小于'ccp_alpha'的子树。默认情况下，不执行修剪。有关详细信息，请参见：ref：'minimal_cost_complexity_pruning',可选浮点数	浮点数	不必须	定数
max_samples	max_samples	默认为None,如果bootstrap为True，则从X抽取以训练每个基本估计量的样本数。-如果没有(默认)，则抽取'X.shape[0]'个样本。-如果为int，则抽取'max_samples'个样本。-如果是浮动的，则抽取'max_samples * X.shape[0]'个样本。因此，'max_samples'应该在'(0, 1)'的区间内,可选整数,浮点数	字符串	不必须	定数

[output]
train_predict	预测	训练集预测结果	一维数组(数值)
test_predict	预测	测试集预测结果	一维数组(数值)
train_score	正确率	训练集预测结果的正确率	数字
test_score	正确率	测试集预测结果的正确率	数字
classes_	类标签	类标签(单输出问题)或类标签数组的列表(多输出问题)	一维数组(数值)
n_classes_	n_classes_	类数(单个输出问题)，或包含每个输出的类数的列表(多输出问题)	整数
feature_importances_	特征重要性	基于杂质的功能的重要性。越高，功能越重要。特征的重要性计算为该特征带来的标准的(归一化)总缩减。这也被称为基尼重要性。警告：基于杂质的特征重要性可能会误导高基数特征(许多唯一值)。参见：func：'sklearn.inspection.permutation_importance'作为替代	一维数组(数值)
n_features_	特征数量	执行'fit'时的功能数量	整数
n_outputs_	输出数量	执行'fit'时的输出数量	整数
oob_score_	oob_score_	使用袋外估计获得的训练数据集的分数。仅当'oob_score'为True时，此属性才存在	浮点数
oob_decision_function_	oob_decision_function_	用训练集上的实际估计值计算的决策函数。如果n_estimators小，则有可能在引导过程中从未遗漏任何数据点。在这种情况下，'oob_decision_function_'可能包含NaN。仅当'oob_score'为True时，此属性才存在	二维数组(数值)

[outline]
额外的树分类器。
此类实现了一种元估计量，该量量估计量适合数据集各个子样本上的许多随机决策树(又称额外树)，并使用求平均值来提高预测准确性和控制过度拟合。

[describe]
额外的树分类器。
此类实现了一种元估计量，该量量估计量适合数据集各个子样本上的许多随机决策树(又称额外树)，并使用求平均值来提高预测准确性和控制过度拟合。

'''


def main(x_train, y_train, x_test, y_test,
         n_estimators=100, criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0., max_features="auto",
         max_leaf_nodes=None, min_impurity_decrease=0., min_impurity_split=None, bootstrap=False, oob_score=False, n_jobs=None, random_state=None, verbose=0,
         warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None
         ):
    if type(x_train) is str:
        x_train = eval(x_train)
    if type(y_train) is str:
        y_train = eval(y_train)
    if type(x_test) is str:
        x_test = eval(x_test)
    if type(y_test) is str:
        y_test = eval(y_test)
    if type(n_estimators) is str:
        n_estimators = eval(n_estimators)
    if type(max_depth) is str:
        max_depth = eval(max_depth)
    if type(min_samples_split) is str:
        min_samples_split = eval(min_samples_split)
    if type(min_samples_leaf) is str:
        min_samples_leaf = eval(min_samples_leaf)
    if type(min_weight_fraction_leaf) is str:
        min_weight_fraction_leaf = eval(min_weight_fraction_leaf)
    if type(max_features) is str and max_features != 'auto' and max_features != 'sqrt' and max_features != 'log2':
        max_features = eval(max_features)
    if type(max_leaf_nodes) is str:
        max_leaf_nodes = eval(max_leaf_nodes)
    if type(min_impurity_decrease) is str:
        min_impurity_decrease = eval(min_impurity_decrease)
    if type(min_impurity_split) is str:
        min_impurity_split = eval(min_impurity_split)
    if type(bootstrap) is str:
        bootstrap = eval(bootstrap)
    if type(oob_score) is str:
        oob_score = eval(oob_score)
    if type(n_jobs) is str:
        n_jobs = eval(n_jobs)
    if type(random_state) is str:
        random_state = eval(random_state)
    if type(verbose) is str:
        verbose = eval(verbose)
    if type(warm_start) is str:
        warm_start = eval(warm_start)
    if type(class_weight) is str:
        class_weight = eval(class_weight)
    if type(ccp_alpha) is str:
        ccp_alpha = eval(ccp_alpha)
    if type(max_samples) is str:
        max_samples = eval(max_samples)

    return r.run(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                 n_estimators=n_estimators,
                 criterion=criterion,
                 max_depth=max_depth,
                 min_samples_split=min_samples_split,
                 min_samples_leaf=min_samples_leaf,
                 min_weight_fraction_leaf=min_weight_fraction_leaf,
                 max_features=max_features,
                 max_leaf_nodes=max_leaf_nodes,
                 min_impurity_decrease=min_impurity_decrease,
                 min_impurity_split=min_impurity_split,
                 bootstrap=bootstrap,
                 oob_score=oob_score,
                 n_jobs=n_jobs,
                 random_state=random_state,
                 verbose=verbose,
                 warm_start=warm_start,
                 class_weight=class_weight,
                 ccp_alpha=ccp_alpha,
                 max_samples=max_samples)


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
