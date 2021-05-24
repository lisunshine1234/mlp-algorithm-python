import numpy as np
import run as  r

'''
[id]
102

[name]
CategoricalNB

[input]
x_train	训练集	训练集标签数据集	二维数组	必须	定数
y_train	测试集	测试集数据集	二维数组	必须	定数
x_test	训练集标签	训练集标签标签	一维数组	必须	定数
y_test	测试集标签	测试集标签	一维数组	必须	定数
alpha	alpha	默认为1.0,加法(拉普拉斯/利兹通)平滑参数(0表示不平滑),可选浮点数	浮点数	不必须	定数
fit_prior	学习先验概率	默认为True,是否学习班级先验概率。如果为假，将使用统一的先验,可选布尔值	布尔值	不必须	定数
class_prior	先验概率	默认为None,该类的先验概率。如果指定，则先验数据不会根据数据进行调整,可选数组	一维数组	不必须	定数

[output]
train_predict	预测	训练集预测结果	一维数组(数值)
test_predict	预测	测试集预测结果	一维数组(数值)
train_score	正确率	训练集预测结果的正确率	数字
test_score	正确率	测试集预测结果的正确率	数字
category_count_	category_count_	为每个特征保存形状的数组(n_classes，各个特征的n_categories)。每个数组提供特定特征的每个类别和类别遇到的样本数量	一维数组
class_count_	类别数	拟合期间每个类别遇到的样本数。提供时，此值由样品重量加权	一维数组
class_log_prior_	类别对数概率	平滑每个类别的经验对数概率	一维数组
classes_	类标签	分类器已知的类标签	一维数组
feature_log_prob_	特征经验对数概率	为每个特征保存形状的数组(n_classes，各个特征的n_categories)。给定各自的特征和类别'P(x_i|y)'，每个数组提供类别的经验对数概率	一维数组
n_features_	特征数量	每个样本的特征数量	整数

[outline]


[describe]
用于分类特征的朴素贝叶斯分类器分类的朴素贝叶斯分类器适用于具有分类分布的离散特征的分类。
每个特征的类别均来自分类分布。

'''


def main(x_train, y_train, x_test, y_test,
         alpha=1.0, fit_prior=True, class_prior=None
         ):
    if type(x_train) is str:
        x_train = eval(x_train)
    if type(y_train) is str:
        y_train = eval(y_train)
    if type(x_test) is str:
        x_test = eval(x_test)
    if type(y_test) is str:
        y_test = eval(y_test)
    if type(alpha) is str:
        alpha = eval(alpha)
    if type(fit_prior) is str:
        fit_prior = eval(fit_prior)
    if type(class_prior) is str:
        class_prior = eval(class_prior)

    return r.run(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, alpha=alpha,
                 fit_prior=fit_prior,
                 class_prior=class_prior)


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