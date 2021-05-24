import run as r


def main(array, row_column, size=None, replace=True, p=None):
    if type(array) is str:
        array = eval(array)
    if type(row_column) is str:
        row_column = eval(row_column)
    if type(size) is str:
        size = eval(size)
    if type(replace) is str:
        replace = eval(replace)
    if type(p) is str:
        p = eval(p)
    return r.run(array=array, row_column=row_column, size=size, replace=replace, p=p)


if __name__ == '__main__':
    import numpy as np
    import json

    array = np.loadtxt('D:\\123_2.csv', delimiter=',')
    # array = array[0:20, :]

    y = array[:, -1].tolist()
    x = np.delete(array, -1, axis=1).tolist()
    array = array.tolist()
    back = main(y, True)

    print(back)
    for i in back:
        print(i + ":" + str(back[i]))

    json.dumps(back)
