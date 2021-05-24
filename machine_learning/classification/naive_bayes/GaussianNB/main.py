import numpy as np
import run as  r

'''
[id]
104

[name]
GaussianNB

[input]
x_train	训练集	训练集标签数据集	二维数组	必须	定数
y_train	测试集	测试集数据集	二维数组	必须	定数
x_test	训练集标签	训练集标签标签	一维数组	必须	定数
y_test	测试集标签	测试集标签	一维数组	必须	定数
priors	先验概率	默认为None,该类的先验概率。如果指定，则先验数据不会根据数据进行调整,可选数组	一维数组	不必须	定数
var_smoothing	征的最大方差	默认为1e-9,所有特征的最大方差部分，已添加到方差中以提高计算稳定性,可选浮点数	浮点数	不必须	定数

[output]
train_predict	预测	训练集预测结果	一维数组(数值)
test_predict	预测	测试集预测结果	一维数组(数值)
train_score	正确率	训练集预测结果的正确率	数字
test_score	正确率	测试集预测结果的正确率	数字
class_count_	类别数	每个课程中观察到的训练样本数	一维数组
class_prior_	class_prior_	每个类的概率	一维数组
classes_	类标签	分类器已知的类标签	一维数组
epsilon_	epsilon_	绝对相加值	浮点数
sigma_	sigma_	每个类别的每个特征的方差	二维数组
theta_	theta_	每类每个特征的平均值	二维数组

[outline]
高斯朴素贝叶斯

[describe]
高斯朴素贝叶斯(GaussianNB)可以通过part_fit进行在线更新模型参数。
有关用于在线更新特征均值和方差的算法的详细信息，请参阅Chan，Golub和LeVeque撰写的Stanford CS技术报告STAN-CS-79-773。

'''


def main(x_train, y_train, x_test, y_test,
         priors=None, var_smoothing=1e-9):
    if type(x_train) is str:
        x_train = eval(x_train)
    if type(y_train) is str:
        y_train = eval(y_train)
    if type(x_test) is str:
        x_test = eval(x_test)
    if type(y_test) is str:
        y_test = eval(y_test)
    if type(priors) is str:
        priors = eval(priors)
    if type(var_smoothing) is str:
        var_smoothing = eval(var_smoothing)

    return r.run(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, priors=priors,
                 var_smoothing=var_smoothing)


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