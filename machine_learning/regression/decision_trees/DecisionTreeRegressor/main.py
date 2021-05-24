import numpy as np
import run as  r

'''
[id]
139

[name]
DecisionTreeRegressor

[input]
x_train	训练集	训练集标签数据集	二维数组	必须	定数
y_train	测试集	测试集数据集	二维数组	必须	定数
x_test	训练集标签	训练集标签标签	一维数组	必须	定数
y_test	测试集标签	测试集标签	一维数组	必须	定数
criterion	criterion	默认为'mse',衡量分割质量的特征。支持的标准是用于均方误差的'mse'，它等于方差减少作为特征选择标准，并使用每个终端节点的均值来最小化L2损失，'friedman_mse'将均方误差与Friedman 's improvement score for potential splits, and ' mae'用作平均绝对误差，使用每个终端节点的中值将L1损耗最小化,可选'friedman_mse','mae','mse'	字符串	不必须	定数
splitter	拆分策略	默认为'best',用于在每个节点上选择拆分的策略。支持的策略是'best'选择最佳拆分和'random'选择最佳随机拆分,可选'random','best'	字符串	不必须	定数
max_depth	最大深度	默认为None,树的最大深度。如果为None，则将节点展开，直到所有叶子都是纯净的，或者直到所有叶子都包含少于min_samples_split个样本,可选整数	整数	不必须	定数
min_samples_split	拆分内部节点最小样本数	默认为2,拆分内部节点所需的最小样本数：。-如果为int，则将'min_samples_split'作为最小数。-如果是浮点型，则'min_samples_split'是分数，而'ceil(min_samples_split * n_samples)'是每个拆分的最小样本数,可选整数,浮点数	字符串	不必须	定数
min_samples_leaf	叶节点最小样本数	默认为1,在叶节点处所需的最小样本数。仅在任何深度的分裂点在左分支和右分支中的每个分支上至少留下'min_samples_leaf'个训练样本时，才考虑。这可能具有平滑模型的效果，尤其是在回归中。-如果为int，则将'min_samples_leaf'作为最小数字。-如果为float，则'min_samples_leaf'是分数，而'ceil(min_samples_leaf * n_samples)'是每个节点的最小样本数,可选整数,浮点数	字符串	不必须	定数
min_weight_fraction_leaf	最小重量分数叶子	默认为0.,在所有叶节点处(所有输入样本)的权重总和中的最小加权分数。如果未提供sample_weight，则样本的权重相等,可选浮点数	浮点数	不必须	定数
max_features	最佳分割特征数量	默认为None,寻找最佳分割时要考虑的特征数量：。-如果为int，则在每个分割中考虑'max_features'个特征。-如果为float，则'max_features'为小数，并且在每个拆分中均考虑'int(max_features * n_features)'特征。-如果是'auto'，则为'max_features=n_features'。-如果是'sqrt'，则为'max_features=sqrt(n_features)'。-如果是'log2'，则为'max_features=log2(n_features)'。-如果为None，则'max_features=n_features'。注意：在找到至少一个有效的节点样本分区之前，分割的搜索不会停止，即使它需要有效地检查'max_features'个以上的特征,可选整数,浮点数,'log2','auto','sqrt'	字符串	不必须	定数
random_state	随机种子	默认为None,控制估算器的随机性。即使将'splitter'设置为'best'，在每次拆分时，这些特征始终都是随机排列的。当为'max_features < n_features'时，算法将在每个拆分中随机选择'max_features'，然后再在其中找到最佳拆分。但是，即使'max_features=n_features'，找到的最佳拆分也可能在不同的运行中有所不同,可选整数	整数	不必须	定数
max_leaf_nodes	最大叶节点	默认为None,以最好的方式用'max_leaf_nodes'种植一棵树。最佳节点定义为杂质的相对减少。如果为None，则叶节点数不受限制,可选整数	整数	不必须	定数
min_impurity_decrease	最小杂质减少	默认为0.,如果节点分裂会导致杂质的减少大于或等于该值，则该节点将被分裂。加权杂质减少方程如下：N_t / N *(杂质-N_t_R / N_t * right_impurity。-N_t_L / N_t * left_impurity)其中'N'是样本总数，'N_t'是当前节点的样本数， 'N_t_L'是左子项中的样本数，'N_t_R'是右子项中的样本数。如果传递了'N'，则'N_t'，'N_t_R'，'N_t_L'和'sample_weight'均指加权总和,可选浮点数	浮点数	不必须	定数
min_impurity_split	最小杂质分裂	默认为None,树木生长尽早停止的阈值。如果节点的杂质高于阈值，则该节点将分裂，否则为叶,可选浮点数	浮点数	不必须	定数
ccp_alpha	复杂性参数	默认为0.0,用于最小化成本复杂性修剪的复杂性参数。将选择成本复杂度最大且小于'ccp_alpha'的子树。默认情况下，不执行修剪。有关详细信息，请参见：ref：'minimal_cost_complexity_pruning',可选浮点数	浮点数	不必须	定数

[output]
train_predict	预测	训练集预测结果	一维数组(数值)
test_predict	预测	测试集预测结果	一维数组(数值)
train_score	正确率	训练集预测结果的正确率	数字
test_score	正确率	测试集预测结果的正确率	数字
feature_importances_	特征重要性	特征的重要性。越高，特征越重要。特征的重要性计算为该特征带来的标准的(标准化)总缩减。它也被称为基尼重要性[4] _。警告：基于杂质的特征重要性可能会误导高基数特征(许多唯一值)。参见：func：'sklearn.inspection.permutation_importance'作为替代	一维数组
max_features_	max_features的推断值	max_features的推断值	整数
n_features_	特征数量	执行'fit'时的特征数量	整数
n_outputs_	输出数量	执行'fit'时的输出数量	整数

[outline]
决策树回归器。

[describe]
决策树回归器。

'''


def main(x_train, y_train, x_test, y_test,
         criterion="mse", splitter="best", max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0., max_features=None,
         random_state=None, max_leaf_nodes=None, min_impurity_decrease=0., min_impurity_split=None, ccp_alpha=0.0
         ):
    if type(x_train) is str:
        x_train = eval(x_train)
    if type(y_train) is str:
        y_train = eval(y_train)
    if type(x_test) is str:
        x_test = eval(x_test)
    if type(y_test) is str:
        y_test = eval(y_test)
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
    if type(random_state) is str:
        random_state = eval(random_state)
    if type(max_leaf_nodes) is str:
        max_leaf_nodes = eval(max_leaf_nodes)
    if type(min_impurity_decrease) is str:
        min_impurity_decrease = eval(min_impurity_decrease)
    if type(min_impurity_split) is str:
        min_impurity_split = eval(min_impurity_split)
    if type(ccp_alpha) is str:
        ccp_alpha = eval(ccp_alpha)

    return r.run(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, criterion=criterion,
                 splitter=splitter,
                 max_depth=max_depth,
                 min_samples_split=min_samples_split,
                 min_samples_leaf=min_samples_leaf,
                 min_weight_fraction_leaf=min_weight_fraction_leaf,
                 max_features=max_features,
                 random_state=random_state,
                 max_leaf_nodes=max_leaf_nodes,
                 min_impurity_decrease=min_impurity_decrease,
                 min_impurity_split=min_impurity_split,
                 ccp_alpha=ccp_alpha)


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
