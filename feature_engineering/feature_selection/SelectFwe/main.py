import numpy as np
import run as r


def main(x, y, score_func='f_classif', alpha=5e-2):

    if type(x) is str:
        x = eval(x)
    if type(y) is str:
        y = eval(y)
    if type(alpha) is str:
        alpha = eval(alpha)
    return r.run(x=x, y=y, score_func=score_func, alpha=alpha)



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