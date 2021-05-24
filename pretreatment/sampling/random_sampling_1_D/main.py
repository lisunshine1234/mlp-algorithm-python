import run as r


def main(array, size=None, replace=True, p=None):
    if type(array) is str:
        array = eval(array)
    if type(p) is str:
        p = eval(p)
    if type(size) is str:
        size = eval(size)
    if type(replace) is str:
        replace = eval(replace)

    return r.run(array=array, size=size, replace=replace, p=p)


if __name__ == '__main__':
    import numpy as np
    import json

    array = np.loadtxt('D:\\123_2.csv', delimiter=',')
    array = array[0:20, :]

    y = array[:, -1].tolist()
    x = np.delete(array, -1, axis=1).tolist()
    array = array.tolist()
    back = main(y)


    print(back)
    for i in back:
        print(i + ":" + str(back[i]))

    json.dumps(back)