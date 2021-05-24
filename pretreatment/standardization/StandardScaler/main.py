import run as r


def main(array, label_index=None, copy=True, with_mean=True, with_std=True):
    if type(array) is str:
        array = eval(array)
    if type(label_index) is str:
        label_index = eval(label_index)
    if type(with_mean) is str:
        with_mean = eval(with_mean)
    if type(with_std) is str:
        with_std = eval(with_std)
    if type(label_index) is str:
        label_index = eval(label_index)
    if type(copy) is str:
        copy = eval(copy)
    return r.run(array=array,
                 label_index=label_index,
                 copy=copy,
                 with_mean=with_mean,
                 with_std=with_std)


if __name__ == '__main__':
    import numpy as np
    import json

    array = np.loadtxt('D:\\123_2.csv', delimiter=',')
    array = array[0:20, :]

    y = array[:, -1].tolist()
    x = np.delete(array, -1, axis=1).tolist()
    array = array.tolist()
    back = main(array,-1)

    print(back)
    for i in back:
        print(i + ":" + str(back[i]))

    json.dumps(back)