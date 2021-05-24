import run as r


def main(array,  classes=None, sparse_output=False):
    if type(array) is str:
        array = eval(array)
    if type(classes) is str:
        classes = eval(classes)
    if type(sparse_output) is str:
        sparse_output = eval(sparse_output)
    return r.run(array=array,
                 classes=classes,
                 sparse_output=sparse_output)


if __name__ == '__main__':
    import numpy as np
    import json

    array = np.loadtxt('D:\\123_2.csv', delimiter=',')
    array = array[0:20, :]

    y = array[:, -1].tolist()
    x = np.delete(array, -1, axis=1).tolist()
    array = array.tolist()
    back = main(array)

    print(back)
    for i in back:
        print(i + ":" + str(back[i]))

    json.dumps(back)