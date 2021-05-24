import numpy as np
import run as  r

'''
[id]
157

[name]
BaggingClassifier

[input]
x_train	训练集	训练集标签数据集	二维数组	必须	定数
y_train	测试集	测试集数据集	二维数组	必须	定数
x_test	训练集标签	训练集标签标签	一维数组	必须	定数
y_test	测试集标签	测试集标签	一维数组	必须	定数
base_estimator	base_estimator	默认为None,基本估计量适合数据集的随机子集。如果为None，则基本估计量为决策树,可选如详情	字符串	不必须	定数
estimator_params	估计器参数	默认为None,参数为默认参数,可选json	json	不必须	定数
n_estimators	n_estimators	默认为10,集合中基本估计量的数量,可选整数	整数	不必须	定数
max_samples	max_samples	默认为1.0,从X抽取以训练每个基本估计量的样本数(默认情况下进行替换，有关更多详细信息，请参见'bootstrap')。-如果为int，则抽取'max_samples'个样本。-如果是浮动的，则抽取'max_samples * X.shape[0]'个样本,可选整数,浮点数	字符串	不必须	定数
max_features	最佳分割特征数量	默认为1.0,从X绘制以训练每个基本估计量的要素数量(默认情况下不进行替换，有关更多详细信息，请参见'bootstrap_features')。-如果为int，则绘制'max_features'功能。-如果为float，则绘制'max_features * X.shape[1]'功能,可选整数,浮点数	字符串	不必须	定数
bootstrap	bootstrap	默认为True,是否抽取样本进行替换。如果为False，则执行不替换的采样,可选布尔值,字符串	字符串	不必须	定数
bootstrap_features	bootstrap_features	默认为False,是否用替换绘制特征,可选布尔值,字符串	字符串	不必须	定数
oob_score	oob_score	默认为False,是否使用现成的样本来估计泛化误差,可选布尔值	布尔值	不必须	定数
warm_start	warm_start	默认为False,设置为True时，请重用上一个调用的解决方案以适合并在集合中添加更多的估计量，否则，就适合,可选布尔值	布尔值	不必须	定数
n_jobs	CPU数量	默认为None,：meth：'fit'和：meth：'predict'并行运行的作业数。 'None'表示1，除非在：obj：'joblib.parallel_backend'上下文中。 '-1'表示全部使用,可选整数	整数	不必须	定数
random_state	随机种子	默认为None,控制原始数据集的随机重采样(明智的抽样和明智的抽样)。如果基本估算器接受'random_state'属性，则为集合中的每个实例生成一个不同的种子。为多个函数调用传递可重复输出的int值,可选整数	整数	不必须	定数
verbose	详细程度	默认为0,在拟合和预测时控制详细程度,可选整数	整数	不必须	定数

[output]
train_predict	预测	训练集预测结果	一维数组(数值)
test_predict	预测	测试集预测结果	一维数组(数值)
train_score	正确率	训练集预测结果的正确率	数字
test_score	正确率	测试集预测结果的正确率	数字
n_features_	特征数量	执行：meth：'fit'时的功能数量	整数
estimators_samples_	estimators_samples_	每个基本估算器的抽取样本(即袋内样本)的子集。每个子集由所选索引的数组定义	二维数组(数值)
estimators_features_	estimators_features_	每个基本估计量的绘制要素子集	二维数组(数值)
classes_	类标签	类标签	一维数组(数值)
n_classes_	n_classes_	类数	整数
oob_score_	oob_score_	使用袋外估计获得的训练数据集的分数。仅当'oob_score'为True时，此属性才存在	浮点数
oob_decision_function_	oob_decision_function_	用训练集上的实际估计值计算的决策函数。如果n_estimators小，则有可能在引导过程中从未遗漏任何数据点。在这种情况下，'oob_decision_function_'可能包含NaN。仅当'oob_score'为True时，此属性才存在	二维数组(数值)

[outline]
套袋分类器。
Bagging分类器是一个集合元估计器，它使每个基本分类器适合原始数据集的随机子集，然后将其单个预测(通过投票或平均)进行汇总以形成最终预测。
通过将随机化引入其构造过程中，然后对其进行整体化，这样的元估计器通常可以用作减少黑盒估计器(例如，决策树)的方差的方式。
该算法涵盖了文献中的几篇著作。
当将数据集的随机子集绘制为样本的随机子集时，该算法称为“粘贴[1] _”。
如果抽取样本进行替换，则该方法称为Bagging [2] _。
当将数据集的随机子集绘制为要素的随机子集时，该方法称为随机子空间[3] _。
最后，当基于样本和特征的子集建立基本估计量时，该方法称为随机补丁[4] _。

[describe]
套袋分类器。
Bagging分类器是一个集合元估计器，它使每个基本分类器适合原始数据集的随机子集，然后将其单个预测(通过投票或平均)进行汇总以形成最终预测。
通过将随机化引入其构造过程中，然后对其进行整体化，这样的元估计器通常可以用作减少黑盒估计器(例如，决策树)的方差的方式。
该算法涵盖了文献中的几篇著作。
当将数据集的随机子集绘制为样本的随机子集时，该算法称为“粘贴[1] _”。
如果抽取样本进行替换，则该方法称为Bagging [2] _。
当将数据集的随机子集绘制为要素的随机子集时，该方法称为随机子空间[3] _。
最后，当基于样本和特征的子集建立基本估计量时，该方法称为随机补丁[4] _。

        'SVC',
        'OneClassSVM',
        'PassiveAggressiveClassifier',
        'Perceptron',
        'RidgeClassifier',
        'RidgeClassifierCV',
        'SGDClassifier',
        'LinearDiscriminantAnalysis',
        'DecisionTreeClassifier',
        'ExtraTreeClassifier',
        'GradientBoostingClassifier',
        'ExtraTreesClassifier',
        'RandomForestClassifier'

'''


def main(x_train, y_train, x_test, y_test,
         base_estimator=None, estimator_params=None,n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False,
         n_jobs=None, random_state=None, verbose=0
         ):
    if type(x_train) is str:
        x_train = eval(x_train)
    if type(y_train) is str:
        y_train = eval(y_train)
    if type(x_test) is str:
        x_test = eval(x_test)
    if type(y_test) is str:
        y_test = eval(y_test)
    if type(estimator_params) is str:
        estimator_params = eval(estimator_params)
    if type(n_estimators) is str:
        n_estimators = eval(n_estimators)
    if type(max_samples) is str:
        max_samples = eval(max_samples)
    if type(max_features) is str:
        max_features = eval(max_features)
    if type(bootstrap) is str:
        bootstrap = eval(bootstrap)
    if type(bootstrap_features) is str:
        bootstrap_features = eval(bootstrap_features)
    if type(oob_score) is str:
        oob_score = eval(oob_score)
    if type(warm_start) is str:
        warm_start = eval(warm_start)
    if type(n_jobs) is str:
        n_jobs = eval(n_jobs)
    if type(random_state) is str:
        random_state = eval(random_state)
    if type(verbose) is str:
        verbose = eval(verbose)

    return r.run(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, base_estimator=base_estimator,
                 n_estimators=n_estimators,
                 max_samples=max_samples,
                 max_features=max_features,
                 bootstrap=bootstrap,
                 bootstrap_features=bootstrap_features,
                 oob_score=oob_score,
                 warm_start=warm_start,
                 n_jobs=n_jobs,
                 random_state=random_state,
                 verbose=verbose,
                 estimator_params=estimator_params
                 )


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
        back = main(x, y, x, y, base_estimator=i)

        print(back)
        for i in back:
            print(i + ":" + str(back[i]))

        json.dumps(back)
