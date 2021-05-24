import numpy as np
import run as r


def main(array, label_index=-1, label_dict=None, replace=True):
    if type(array) is str:
        array = eval(array)
    if type(label_index) is str:
        label_index = eval(label_index)
    if type(label_dict) is str:
        label_dict = eval(label_dict)
    if type(replace) is str:
        replace = eval(replace)
    return r.run(array=array, label_index=label_index, label_dict=label_dict, replace=replace)


if __name__ == '__main__':
    import numpy as np
    import json

    array = np.loadtxt('D:\\123_2.csv', delimiter=',')
    array = array[0:20, :]

    y = array[:, -1].tolist()
    x = np.delete(array, -1, axis=1).tolist()
    array = array.tolist()
    back = main(array, -1)

    print(back)
    for i in back:
        print(i + ":" + str(back[i]))

    json.dumps(back)
