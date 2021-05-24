import numpy as np
import run as  r

'''
[id]
154

[name]
GradientBoostingRegressor

[input]
x_train	训练集	训练集标签数据集	二维数组	必须	定数
y_train	测试集	测试集数据集	二维数组	必须	定数
x_test	训练集标签	训练集标签标签	一维数组	必须	定数
y_test	测试集标签	测试集标签	一维数组	必须	定数
loss	损失函数	默认为ls,损失函数有待优化。 'ls'表示最小二乘回归。 'lad'(最小绝对偏差)是仅基于输入变量的顺序信息的高度鲁棒的损失函数。 'huber'是两者的组合。 'quantile'允许分位数回归(使用'alpha'指定分位数),可选'quantile','lad','huber','ls'	字符串	不必须	定数
learning_rate	学习率	默认为0.1,学习率使每棵树的贡献减少'learning_rate'。在learning_rate和n_estimators之间需要权衡,可选浮点数	浮点数	不必须	定数
n_estimators	n_estimators	默认为100,要执行的提升阶段数。梯度提升对于过度拟合具有相当强的鲁棒性，因此大量提升通常会带来更好的性能,可选整数	整数	不必须	定数
subsample	subsample	默认为1.0,用于拟合各个基础学习者的样本比例。如果小于1.0，则将导致随机梯度增强。 'subsample'与参数'n_estimators'交互。选择'subsample < 1.0'会导致方差减少和偏差增加,可选浮点数	浮点数	不必须	定数
criterion	标准	默认为friedman_mse,衡量分割质量的功能。支持的标准是：'friedman_mse'表示具有弗里德曼改进得分的均方误差，'mse'表示均方误差，以及'mae'表示平均绝对误差。 'friedman_mse'的默认值通常是最好的，因为在某些情况下它可以提供更好的近似值,可选'mse','friedman_mse','mae'	字符串	不必须	定数
min_samples_split	拆分最小样本数	默认为2,拆分内部节点所需的最小样本数：。-如果为int，则将'min_samples_split'作为最小数。-如果是浮点型，则'min_samples_split'是分数，而'ceil(min_samples_split * n_samples)'是每个拆分的最小样本数,可选整数,浮点数	字符串	不必须	定数
min_samples_leaf	最小样本数	默认为1,在叶节点处所需的最小样本数。仅在任何深度的分裂点在左分支和右分支中的每个分支上至少留下'min_samples_leaf'个训练样本时，才考。-如果为int，则将'min_samples_leaf'作为最小数字。-如果为float，则'min_samples_leaf'是分数，而'ceil(min_samples_leaf * n_samples)'是每个节点的最小样本数,可选整数,浮点数	字符串	不必须	定数
min_weight_fraction_leaf	最小重量分数叶子	默认为0.,在所有叶节点处(所有输入样本)的权重总和中的最小加权分数。如果未提供sample_weight，则样本的权重相等,可选浮点数	浮点数	不必须	定数
max_depth	最大深度	默认为3,各个回归估计量的最大深度。最大深度限制了树中节点的数量。调整此参数以获得最佳性能；最佳值取决于输入变量的相互作用,可选整数	整数	不必须	定数
min_impurity_decrease	最小杂质减少	默认为0.,如果节点分裂会导致杂质的减少大于或等于该值，则该节点将被分裂。加权杂质减少方程如下：N_t / N *(杂质-N_t_R / N_t * right_impurity。-N_t_L / N_t * left_impurity)其中'N'是样本总数，'N_t'是当前节点的样本数， 'N_t_L'是左子项中的样本数，'N_t_R'是右子项中的样本数。如果传递了'N'，则'N_t'，'N_t_R'，'N_t_L'和'sample_weight'均指加权总和,可选浮点数	浮点数	不必须	定数
min_impurity_split	最小杂质分裂	默认为None,树木生长尽早停止的阈值。如果节点的杂质高于阈值，则该节点将分裂，否则为叶,可选浮点数	浮点数	不必须	定数
init	初始化方法	默认为None,一个估计器对象，用于计算初始预测。 'init'必须提供：term：'fit'和：term：'predict'。如果为'zero'，则初始原始预测设置为零。默认情况下，使用'DummyEstimator'来预测平均目标值(亏损= 'ls')或其他损失的分位数,可选'zero'	字符串	不必须	定数
estimator_params	估计器参数	默认为None,参数为默认参数,可选json	json	不必须	定数
random_state	随机种子	默认为None,控制每次增强迭代时分配给每个Tree估计量的随机种子。另外，它控制每个分割处特征的随机排列(有关更多详细信息，请参见注释)。如果'n_iter_no_change'不为None，它还控制训练数据的随机拆分以获得验证集。为多个函数调用传递可重复输出的int值,可选整数	整数	不必须	定数
max_features	最佳分割特征数量	默认为None,寻找最佳分割时要考虑的要素数量：-如果为int，则在每个分割中考虑'max_features'个要素。-如果为float，则'max_features'为小数。-如果为'auto'，则为'max_features=n_features'。-如果为'sqrt'，则为'max_features=sqrt(n_features)'。-如果为'log2'，则为'max_features=log2(n_features)'。-如果为None，则'max_features=n_features',可选整数,浮点数,'auto','sqrt','log2'	字符串	不必须	定数
alpha	alpha	默认为0.9,Huber损失函数和分位数损失函数的alpha分位数。仅当'loss=' huber ' or ' loss = 'quantile'时,可选浮点数	浮点数	不必须	定数
verbose	详细程度	默认为0,启用详细输出。如果为1，则偶尔打印一次进度和性能(树越多，频率越低)。如果大于1，则为每棵树打印进度和性能,可选整数	整数	不必须	定数
max_leaf_nodes	最大叶节点	默认为None,以'max_leaf_nodes'优先的方式种植$。最佳节点定义为杂质的相对减少。如果为None，则叶节点数不受限制,可选整数	整数	不必须	定数
warm_start	warm_start	默认为False,设置为'True'时，请重用上一个调用的解决方案以适应并在集合中添加更多估算器，否则，只需擦除,可选布尔值	布尔值	不必须	定数
presort	presort	默认为'deprecated',此参数已弃用，并将在v0.24中删除,可选'deprecated'	字符串	不必须	定数
validation_fraction	验证分数	默认为0.1,预留的训练数据比例作为早期停止的验证集。必须在0到1之间。仅在'n_iter_no_change'设置为整数时使用,可选浮点数	浮点数	不必须	定数
n_iter_no_change	迭代不变次数	默认为None,'n_iter_no_change'用于确定在验证分数没有改善时是否将使用提前停止来终止训练。默认情况下，将其设置为None以禁用提前停止。如果设置为数字，则它将保留训练数据的大小作为验证，并在之前的所有'validation_fraction'迭代次数中验证得分均未提高时终止训练,可选整数	整数	不必须	定数
tol	tol	默认为1e-4,允许提前停止。如果在'n_iter_no_change'次迭代中损失至少没有改善(如果设置为数字)，则训练停止,可选浮点数	浮点数	不必须	定数
ccp_alpha	复杂性参数	默认为0.0,用于最小成本复杂性修剪的复杂性参数。将选择成本复杂度最大且小于'ccp_alpha'的子树。默认情况下，不执行修剪。有关详细信息，请参见：ref：'minimal_cost_complexity_pruning',可选浮点数	浮点数	不必须	定数

[output]
train_predict	预测	训练集预测结果	一维数组(数值)
test_predict	预测	测试集预测结果	一维数组(数值)
train_score	正确率	训练集预测结果的正确率	数字
test_score	正确率	测试集预测结果的正确率	数字
feature_importances_	特征重要性	基于杂质的功能的重要性。越高，功能越重要。特征的重要性计算为该特征带来的标准的(归一化)总缩减。这也被称为基尼重要性。警告：基于杂质的特征重要性可能会误导高基数特征(许多唯一值)。参见：func：'sklearn.inspection.permutation_importance'作为替代	一维数组(数值)
oob_improvement_	oob_improvement_	相对于先前的迭代，袋装样本的损失(=偏差)的改善。 'oob_improvement_[0]'是第一阶段损失比'init'估算器的改善。仅在'subsample < 1.0'时可用	一维数组(数值)
train_score_	train_score_	第i个得分'train_score_[i]'是袋内样本在迭代'i'处模型的偏差(=损失)。如果'subsample == 1'，则表示训练数据存在偏差	一维数组(数值)
n_features_	特征数量	数据功能的数量	整数
max_features_	先验概率	max_features的推断值	整数

[outline]
梯度提升以进行回归。
GB以渐进的阶段方式建立加性模型；它允许优化任意微分损失函数。
在每个阶段，将回归树拟合到给定损失函数的负梯度上。

[describe]
梯度提升以进行回归。
GB以渐进的阶段方式建立加性模型；它允许优化任意微分损失函数。
在每个阶段，将回归树拟合到给定损失函数的负梯度上。
'SVR',
        'SVC',
        'NuSVR',
        'OneClassSVM',
        'ARDRegression',
        'BayesianRidge',
        'ElasticNet',
        'ElasticNetCV',
        'HuberRegressor',
        'Lars',
        'LarsCV',
        'Lasso',
        'LassoCV',
        'LassoLars',
        'LassoLarsCV',
        'LassoLarsIC',
        'LinearRegression',
        'OrthogonalMatchingPursuit',
        'OrthogonalMatchingPursuitCV',
        'PassiveAggressiveClassifier',
        'PassiveAggressiveRegressor',
        'Perceptron',
        'Ridge',
        'RidgeClassifier',
        'RidgeClassifierCV',
        'RidgeCV',
        'SGDClassifier',
        'SGDRegressor',
        'TheilSenRegressor',
        'LinearDiscriminantAnalysis',
        'DecisionTreeClassifier',
        'DecisionTreeRegressor',
        'ExtraTreeClassifier',
        'ExtraTreeRegressor',
        'GradientBoostingClassifier',
        'ExtraTreesClassifier',
        'RandomForestClassifier'
'''


def main(x_train, y_train, x_test, y_test,
         loss='ls', estimator_params=None, learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2,
         min_samples_leaf=1,
         min_weight_fraction_leaf=0., max_depth=3, min_impurity_decrease=0., min_impurity_split=None, init=None, random_state=None, max_features=None,
         alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, validation_fraction=0.1, n_iter_no_change=None, tol=1e-4, ccp_alpha=0.0
         ):
    if type(x_train) is str:
        x_train = eval(x_train)
    if type(y_train) is str:
        y_train = eval(y_train)
    if type(x_test) is str:
        x_test = eval(x_test)
    if type(y_test) is str:
        y_test = eval(y_test)
    if type(learning_rate) is str:
        learning_rate = eval(learning_rate)
    if type(n_estimators) is str:
        n_estimators = eval(n_estimators)
    if type(subsample) is str:
        subsample = eval(subsample)
    if type(min_samples_split) is str:
        min_samples_split = eval(min_samples_split)
    if type(min_samples_leaf) is str:
        min_samples_leaf = eval(min_samples_leaf)
    if type(min_weight_fraction_leaf) is str:
        min_weight_fraction_leaf = eval(min_weight_fraction_leaf)
    if type(max_depth) is str:
        max_depth = eval(max_depth)
    if type(min_impurity_decrease) is str:
        min_impurity_decrease = eval(min_impurity_decrease)
    if type(min_impurity_split) is str:
        min_impurity_split = eval(min_impurity_split)
    if type(estimator_params) is str:
        estimator_params = eval(estimator_params)
    if type(random_state) is str:
        random_state = eval(random_state)
    if type(max_features) is str and max_features != 'auto' and max_features != 'sqrt' and max_features != 'log2':
        max_features = eval(max_features)
    if type(alpha) is str:
        alpha = eval(alpha)
    if type(verbose) is str:
        verbose = eval(verbose)
    if type(max_leaf_nodes) is str:
        max_leaf_nodes = eval(max_leaf_nodes)
    if type(warm_start) is str:
        warm_start = eval(warm_start)
    if type(validation_fraction) is str:
        validation_fraction = eval(validation_fraction)
    if type(n_iter_no_change) is str:
        n_iter_no_change = eval(n_iter_no_change)
    if type(tol) is str:
        tol = eval(tol)
    if type(ccp_alpha) is str:
        ccp_alpha = eval(ccp_alpha)

    return r.run(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, loss=loss,
                 learning_rate=learning_rate,
                 n_estimators=n_estimators,
                 subsample=subsample,
                 criterion=criterion,
                 min_samples_split=min_samples_split,
                 min_samples_leaf=min_samples_leaf,
                 min_weight_fraction_leaf=min_weight_fraction_leaf,
                 max_depth=max_depth,
                 min_impurity_decrease=min_impurity_decrease,
                 min_impurity_split=min_impurity_split,
                 init=init,
                 estimator_params=estimator_params,
                 random_state=random_state,
                 max_features=max_features,
                 alpha=alpha,
                 verbose=verbose,
                 max_leaf_nodes=max_leaf_nodes,
                 warm_start=warm_start,
                 validation_fraction=validation_fraction,
                 n_iter_no_change=n_iter_no_change,
                 tol=tol,
                 ccp_alpha=ccp_alpha)


if __name__ == '__main__':
    import numpy as np
    import json

    array = np.loadtxt('D:\\123_2.csv', delimiter=',')
    array = array[0:20, :]

    y = array[:, -1].tolist()
    x = np.delete(array, -1, axis=1).tolist()
    array = array.tolist()
    for i in [
        'SVR',
        'SVC',
        'NuSVR',
        'OneClassSVM',
        'ARDRegression',
        'BayesianRidge',
        'ElasticNet',
        'ElasticNetCV',
        'HuberRegressor',
        'Lars',
        'LarsCV',
        'Lasso',
        'LassoCV',
        'LassoLars',
        'LassoLarsCV',
        'LassoLarsIC',
        'LinearRegression',
        'OrthogonalMatchingPursuit',
        'OrthogonalMatchingPursuitCV',
        'PassiveAggressiveClassifier',
        'PassiveAggressiveRegressor',
        'Perceptron',
        'Ridge',
        'RidgeClassifier',
        'RidgeClassifierCV',
        'RidgeCV',
        'SGDClassifier',
        'SGDRegressor',
        'TheilSenRegressor',
        'LinearDiscriminantAnalysis',
        'DecisionTreeClassifier',
        'DecisionTreeRegressor',
        'ExtraTreeClassifier',
        'ExtraTreeRegressor',
        'GradientBoostingClassifier',
        'ExtraTreesClassifier',
        'RandomForestClassifier'
    ]:
        back = main(x, y, x, y, init=i)

        print(back)
        for i in back:
            print(i + ":" + str(back[i]))

        json.dumps(back)
