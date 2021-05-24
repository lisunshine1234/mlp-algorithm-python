import numpy as np
import run as r
'''
[id]
89

[name]
DictionaryLearning

[input]
array 数据集 数据集 二维数组 必须 定数
label_index 标签列号 默认为None,表示所有列参与训练，输入整数代表标签所在的列号，可选整数 数字 不必须 定数
n_components 字典元素数 默认为n_features,要提取的字典元素数，可选整数 数字 不必须 定数
alpha 稀疏控制参数 默认为1.0,稀疏控制参数，可选浮点数 浮点数 不必须 定数
max_iter 最大迭代次数 默认为1000,要执行的最大迭代次数，可选整数 整数 不必须 定数
tol 数值误差公差 默认为0.00000001,数值误差公差，可选浮点数 浮点数 不必须 定数
fit_algorithm 适应算法 默认为lars,lars：使用最小角度回归方法来解决套索问题;cd：使用坐标下降法来计算套索解决方案。如果估计的分量稀疏，Lars将更快，可选'lars','cd' 字符串 不必须 定数
transform_algorithm 转换算法 默认为omp；用于转换数据的算法lars：使用最小角度回归方法；lasso_lars：使用Lars计算套索解决方案；lasso_cd：使用坐标下降法计算套索解决方案。如果估计的组件稀疏，则lasso_lars会更快；omp：使用正交匹配追踪来估计稀疏解阈值：将投影dictionary*X中小于alpha的所有系数都压缩为零；可选lasso_lars,lasso_cd,lars,omp,threshold 字符串 不必须 定数
transform_n_nonzero_coefs 非零系数的数量 默认为0.1*n_features,在解决方案的每一列中定位的非零系数的数量。仅由`algorithm='lars'和`algorithm='omp'`使用，在`omp`情况下被`alpha`覆盖，可选整数 整数 不必须 定数
transform_alpha 转换alpha 默认为1.0,如果`algorithm='lasso_lars'`或`algorithm='lasso_cd'`，则“alpha”是应用于L1范数的惩罚。如果'algorithm='threshold'`，则'alpha'是阈值的绝对值，低于该阈值时系数将被压缩为零。如果“algorithm='omp'”，则“alpha”是公差参数：目标为重建误差的值。在这种情况下，它将覆盖`n_nonzero_coefs`，可选浮点数 浮点数 不必须 定数
n_jobs 并行作业数 默认为None,要运行的并行作业数。’None‘表示1。-1表示使用所有处理器，可选整数 整数 不必须 定数
code_init 代码初始值 默认为None,代码的初始值，用于热启动，可选二维数组(n_samples,n_components) 二维数组 不必须 定数
dict_init 字典初始值 默认为None,字典的初始值，用于热启动，可选二维数组(n_samples,n_components) 二维数组 不必须 定数
verbose 控制过程的详细程度 默认为False,控制过程的详细程度 布尔值 不必须 定数
split_sign 稀疏特征向量串联 默认为False,是否将稀疏特征向量分为其负部分和正部分的串联。这样可以提高下游分类器的性能 布尔值 不必须 定数
random_state 随机状态 默认为None,未指定dict_init时，用于初始化字典；将shuffle设置为True时，对数据随机改组，并更新字典。在多个函数调用之间传递int以获得可重复的结果，可选整数 整数 不必须 定数
positive_code 代码积极性 默认为False,在寻找代码时是否加强积极性 布尔值 不必须 定数
positive_dict 字典积极性 默认为False,查找字典时是否要加强积极性 布尔值 不必须 定数
transform_max_iter 字典积极性 默认为1000,如果`algorithm='lasso_cd'`或`lasso_lars`时要执行的最大迭代次数，可选整数 整数 不必须 定数

[output]
array 数组 训练之后的数组 二维数组
components_ 字典原子 从数据中提取字典原子 二维数组
error_ 误差向量 每次迭代的误差向量 二维数组
n_iter_ 迭代次数 运行的迭代次数 数字

[outline]
查找最适合用来使用稀疏代码表示数据的字典（一组原子）。

[describe]
查找最适合用来使用稀疏代码表示数据的字典（一组原子）。
解决优化问题：
（U ^ *，V ^ *）= argmin 0.5 || Y-U V || _2 ^ 2 + alpha * || U || _1（U，V）与|| V_k || _2= 1，
所有0 <= k <n_components
'''


def main(array, label_index=None, n_components=None, alpha=1, max_iter=1000, tol=1e-8, fit_algorithm='lars',
         transform_algorithm='omp', transform_n_nonzero_coefs=None, transform_alpha=None, n_jobs=None, code_init=None,
         dict_init=None, verbose=False, split_sign=False, random_state=None, positive_code=False, positive_dict=False,
         transform_max_iter=1000):
    if type(array) is str:
        array = eval(array)
    if type(label_index) is str:
        label_index = eval(label_index)
    if type(n_components) is str:
        n_components = eval(n_components)
    if type(alpha) is str:
        alpha = eval(alpha)
    if type(max_iter) is str:
        max_iter = eval(max_iter)
    if type(tol) is str:
        tol = eval(tol)
    if type(transform_n_nonzero_coefs) is str:
        transform_n_nonzero_coefs = eval(transform_n_nonzero_coefs)
    if type(transform_alpha) is str:
        transform_alpha = eval(transform_alpha)
    if type(n_jobs) is str:
        n_jobs = eval(n_jobs)
    if type(code_init) is str:
        code_init = eval(code_init)
    if type(dict_init) is str:
        dict_init = eval(dict_init)
    if type(verbose) is str:
        verbose = eval(verbose)
    if type(split_sign) is str:
        split_sign = eval(split_sign)
    if type(random_state) is str:
        random_state = eval(random_state)
    if type(positive_code) is str:
        positive_code = eval(positive_code)
    if type(positive_dict) is str:
        positive_dict = eval(positive_dict)
    if type(transform_max_iter) is str:
        transform_max_iter = eval(transform_max_iter)

    return r.run(array=array,
                 label_index=label_index,
                 n_components=n_components,
                 alpha=alpha,
                 max_iter=max_iter,
                 tol=tol,
                 fit_algorithm=fit_algorithm,
                 transform_algorithm=transform_algorithm,
                 transform_n_nonzero_coefs=transform_n_nonzero_coefs,
                 transform_alpha=transform_alpha,
                 n_jobs=n_jobs,
                 code_init=code_init,
                 dict_init=dict_init,
                 verbose=verbose,
                 split_sign=split_sign,
                 random_state=random_state,
                 positive_code=positive_code,
                 positive_dict=positive_dict,
                 transform_max_iter=transform_max_iter)


if __name__ == '__main__':
    import numpy as np
    import json
    array = np.loadtxt('D:\\123_2.csv', delimiter=',')
    # y= array[:, -1]
    # x= np.delete(array, -1, axis=1)
    #
    # x= x.tolist()
    # y= y.tolist()
    array = array.tolist()
    print(main(array,n_components=2,n_jobs=1))
