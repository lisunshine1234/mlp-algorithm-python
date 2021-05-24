import numpy as np
import run as  r

'''
[id]
151

[name]
AdaBoostRegressor

[input]
x_train	训练集	训练集标签数据集	二维数组	必须	定数
y_train	测试集	测试集数据集	二维数组	必须	定数
x_test	训练集标签	训练集标签标签	一维数组	必须	定数
y_test	测试集标签	测试集标签	一维数组	必须	定数
base_estimator	base_estimator	默认为None,从中构建增强合奏的基本估计量。如果为'None'，则基本估计量为'DecisionTreeRegressor(max_depth=3)',可选RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor	字符串	不必须	定数
estimator_params	估计器参数	默认为None,参数为默认参数,可选json	json	不必须	定数
n_estimators	n_estimators	默认为50,终止增强的最大估计量。在完美配合的情况下，学习程序将尽早停止,可选整数	整数	不必须	定数
learning_rate	学习率	默认为1.,学习率使每个回归变量的贡献减少'learning_rate'。 'learning_rate'和'n_estimators'之间需要权衡,可选浮点数	浮点数	不必须	定数
loss	损失函数	默认为linear,每次增强迭代后更新权重时使用的损失函数,可选'square','exponential','linear'	字符串	不必须	定数
random_state	随机种子	默认为None,控制每次增强迭代中每个'base_estimator'给出的随机种子。因此，仅在'base_estimator'公开'random_state'时使用。此外，它还控制权重的自举，该权重用于在每次提升迭代中训练'base_estimator'。为多个函数调用传递可重复输出的int值,可选整数	整数	不必须	定数

[output]
train_predict	预测	训练集预测结果	一维数组(数值)
test_predict	预测	测试集预测结果	一维数组(数值)
train_score	正确率	训练集预测结果的正确率	数字
test_score	正确率	测试集预测结果的正确率	数字
estimator_weights_	estimator_weights_	增强合奏中每个估计量的权重	一维数组(数值)
estimator_errors_	estimator_errors_	增强集合中每个估计量的回归误差	一维数组(数值)
feature_importances_	特征重要性	如果有'base_estimator'支持，则基于杂质的特征的重要性(当基于决策树时)	一维数组(数值)

[outline]
AdaBoost回归器。
AdaBoost回归器是一种元估计器，它首先将回归器拟合到原始数据集上，然后将回归器的其他副本拟合到同一数据集上，但根据当前预测的误差调整实例的权重。
因此，随后的回归者更多地关注困难情况。
此类实现称为AdaBoost.R2 [2]的算法。

[describe]
AdaBoost回归器。
AdaBoost回归器是一种元估计器，它首先将回归器拟合到原始数据集上，然后将回归器的其他副本拟合到同一数据集上，但根据当前预测的误差调整实例的权重。
因此，随后的回归者更多地关注困难情况。
此类实现称为AdaBoost.R2 [2]的算法。

'''


def main(x_train, y_train, x_test, y_test,
         base_estimator=None, estimator_params=None, n_estimators=50, learning_rate=1., loss='linear', random_state=None
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
    if type(learning_rate) is str:
        learning_rate = eval(learning_rate)
    if type(random_state) is str:
        random_state = eval(random_state)
    if type(estimator_params) is str:
        estimator_params = eval(estimator_params)
    return r.run(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                 base_estimator=base_estimator,
                 n_estimators=n_estimators,
                 learning_rate=learning_rate,
                 loss=loss,
                 random_state=random_state,
                 estimator_params=estimator_params)


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
