import numpy as np
import run as  r

'''
[id]
124

[name]
LogisticRegression

[input]
x_train	训练集	训练集标签数据集	二维数组	必须	定数
y_train	测试集	测试集数据集	二维数组	必须	定数
x_test	训练集标签	训练集标签标签	一维数组	必须	定数
y_test	测试集标签	测试集标签	一维数组	必须	定数
penalty	惩罚规范	默认为'l2',用于指定惩罚中使用的规范。newton-cg，sag和lbfgs求解器仅支持l2罚分。仅saga求解器支持elasticnet。如果为none(liblinear求解器不支持)，则不应用任何正则化。使用SAGA解算器的罚款为0.19l1(允许多项式+L1),可选l1,l2,elasticnet,none	字符串	不必须	定数
dual	dual	默认为False,对偶或原始配方。对偶公式化仅使用liblinear求解器实现l2惩罚。当n_samples>n_features时，首选dual=False,可选布尔值	布尔值	不必须	定数
tol	tol	默认为1e-4,停止标准的公差,可选浮点数	浮点数	不必须	定数
C	C	默认为1.0,正则强度的倒数；必须为正浮点数。像在支持向量机中一样，较小的值指定更强的正则化,可选浮点数	浮点数	不必须	定数
fit_intercept	计算截距	默认为True,指定是否应将常量(也称为偏差或截距)添加到决策函数,可选整数,布尔值	字符串	不必须	定数
intercept_scaling	截距缩放	默认为1,仅在使用求解器liblinear并将self.fit_intercept设置为True时有用。在这种情况下，x变为[x，self.intercept_scaling]，即，将常量值等于intercept_scaling的合成特征附加到实例向量。截距变为intercept_scaling*Composite_feature_weight,可选整数,浮点数	字符串	不必须	定数
class_weight	类别权重	默认为None,与类别相关的权重，格式为{class_label：weight}。如果未给出，则所有类都应具有权重一。平衡模式使用y的值自动将权重与输入数据中的类频率成反比地调整为n_samples/(n_classes*np.bincount(y))。请注意，如果指定了sample_weight，则这些权重将与sample_weight(通过fit方法传递)相乘。,可选字典，字符串	字符串	不必须	定数
random_state	随机状态	默认为None,在solver==sag，saga或liblinear洗牌时使用,可选整数	整数	不必须	定数
solver	solver	默认为lbfgs,用于优化问题的default=lbfgs算法。对于小型数据集，liblinear是一个不错的选择，而对于大型数据集，sag和saga则更快。对于多类问题，只有newtoncg，sag，saga和lbfgs处理多项式损失；liblinear仅限于一站式计划。newton-cg，lbfgs，sag和saga处理L2或不处罚liblinear和saga也处理L1处罚;saga还支持elasticnet处罚;liblinear可以不支持设置penalty=none,可选newton-cg,lbfgs,liblinear,sag,saga	字符串	不必须	定数
max_iter	最大迭代次数	默认为100,求解程序收敛所需的最大迭代次数,可选整数	整数	不必须	定数
multi_class	多类别	默认为'auto',如果选择的选项是ovr，则每个标签都适合二进制问题。对于多项式，最小化的损失是整个概率分布上的多项式损失拟合，即使数据是二进制的，也是如此。当solver=liblinear时，multinomial不可用。如果数据是二进制的，或者如果Solver=liblinear，则auto选择ovr，否则选择multinomial。可选auto,ovr,multinomial	字符串	不必须	定数
verbose	详细程度	默认为0,对于liblinear和lbfgs，求解器将verbose设置为任何正数以表示详细程度,可选整数	整数	不必须	定数
warm_start	warm_start	默认为False,设置为True时，请重用上一个调用的解决方案以适合初始化，否则，只需擦除以前的解决方案即可,可选布尔值	布尔值	不必须	定数
n_jobs	CPU数量	默认为None,如果multi_class=ovr，则在对类进行并行化时使用的CPU内核数。当solver设置为liblinear时，无论是否指定了multi_class，该参数都将被忽略。除非在：obj：joblib.parallel_backend上下文中，否则表示1，.-1表示使用所有处理器,可选整数	整数	不必须	定数
l1_ratio	l1_ratio	默认为None,Elastic-Net混合参数，其值为0<=l1_ratio<=1。仅在penalty=elasticnet时使用。设置l1_ratio=0等同于使用penalty=l2，而设置l1_ratio=1等同于使用penalty=l1。对于0<l1_ratio<1，惩罚是L1和L2的组合,可选浮点数	浮点数	不必须	定数

[output]
train_predict	预测	训练集预测结果	一维数组(数值)
test_predict	预测	测试集预测结果	一维数组(数值)
train_score	正确率	训练集预测结果的正确率	数字
test_score	正确率	测试集预测结果的正确率	数字
classes_	classes_	无描述信息	二维数组
coef_	参数向量	分类器已知的类别标签列表	二维数组
intercept_	截距	决策函数中特征的系数。当给定问题为二进制时，coef_的形状为(1，n_features)。特别地，当multi_class=multinomial时，coef_对应于结果1(真)，而-coef_对应于结果0(假)	整数
n_iter_	迭代次数	拦截(又称偏差)已添加到决策特征。如果fit_intercept设置为False，则截距设置为零。当给定问题为二进制时，intercept_的形状为(1，)。特别地，当multi_class=multinomial时，intercept_对应于结果1(真)，而-intercept_对应于结果0(假)	一维数组

[outline]


[describe]
Logistic回归(又名logit，MaxEnt)分类器。
在多类情况下，训练算法使用一比休息(OvR)如果multi_class选项设置为ovr，并使用如果multi_class选项设置为multinomial，则交叉熵损失。
(当前，多项式选项仅受lbfgs支持，sag，saga和newton-cg解算器。
)此类使用liblinear库，newton-cg，sag，saga和lbfgs求解器。
**注意默认情况下应用正则化**。
它既可以处理密集和稀疏输入。
使用包含64位的C排序数组或CSR矩阵浮动以获得最佳性能；任何其他输入格式将被转换(并复制)。
newton-cg，sag和lbfgs求解器仅支持L2正则化具有原始公式，或者没有正则化。
liblinear求解器支持L1和L2正则化，仅对L2罚款。
Elastic-Net正则化仅受saga求解器
'''


def main(x_train, y_train, x_test, y_test,
         penalty='l2', dual=False, tol=1e-4, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None,
         random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None,
         l1_ratio=None
         ):
    if type(x_train) is str:
        x_train = eval(x_train)
    if type(y_train) is str:
        y_train = eval(y_train)
    if type(x_test) is str:
        x_test = eval(x_test)
    if type(y_test) is str:
        y_test = eval(y_test)
    if type(dual) is str:
        dual = eval(dual)
    if type(tol) is str:
        tol = eval(tol)
    if type(C) is str:
        C = eval(C)
    if type(fit_intercept) is str:
        fit_intercept = eval(fit_intercept)
    if type(intercept_scaling) is str:
        intercept_scaling = eval(intercept_scaling)
    if type(class_weight) is str and class_weight != 'balanced':
        class_weight = eval(class_weight)
    if type(random_state) is str:
        random_state = eval(random_state)
    if type(max_iter) is str:
        max_iter = eval(max_iter)
    if type(verbose) is str:
        verbose = eval(verbose)
    if type(warm_start) is str:
        warm_start = eval(warm_start)
    if type(n_jobs) is str:
        n_jobs = eval(n_jobs)
    if type(l1_ratio) is str:
        l1_ratio = eval(l1_ratio)

    return r.run(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, penalty=penalty,
                 dual=dual,
                 tol=tol,
                 C=C,
                 fit_intercept=fit_intercept,
                 intercept_scaling=intercept_scaling,
                 class_weight=class_weight,
                 random_state=random_state,
                 solver=solver,
                 max_iter=max_iter,
                 multi_class=multi_class,
                 verbose=verbose,
                 warm_start=warm_start,
                 n_jobs=n_jobs,
                 l1_ratio=l1_ratio)



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
