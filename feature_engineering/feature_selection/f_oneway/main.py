import numpy as np
import run as r

'''
[id]
86

[name]
f_oneway

[input]
args 样本测量值 sample1，sample2 ...样本测量值应作为参数给出 不定数组  必须 不定数

[output]
F-value F值 计算的测试F值 一维数组
p-value p值 F分布中的关联p值。 一维数组

[outline]
单变量线性回归测试。

[describe]
进行单因素方差分析
在单因素方差分析检验零假设，即2个或多个组具有相同的总体均值。
 该测试是从两个或更多个基团施加到样品，可能具有不同的尺寸
'''
def main(*args):
    arg_temp=[]
    for arg in args:
        if type(arg) is str:
            arg_temp.append(eval(arg))
        else:
            arg_temp.append(arg)
    return r.run(*arg_temp)


if __name__ == '__main__':
    import numpy as np
    import json

    array = np.loadtxt('D:\\123_2.csv', delimiter=',')
    array = array[0:20, :]

    y = array[:, -1].tolist()
    x = np.delete(array, -1, axis=1).tolist()
    array = array.tolist()
    back = main(x, y)

    print(back)
    for i in back:
        print(i + ":" + str(back[i]))

    json.dumps(back)

