import run as r


def main(array, label_index=None, with_centering=True, with_scaling=True,
         quantile_range=(25.0, 75.0), copy=True):
    if type(array) is str:
        array = eval(array)
    if type(label_index) is str:
        label_index = eval(label_index)
    if type(with_centering) is str:
        with_centering = eval(with_centering)
    if type(with_scaling) is str:
        with_scaling = eval(with_scaling)
    if type(quantile_range) is str:
        quantile_range = eval(quantile_range)
    if type(label_index) is str:
        label_index = eval(label_index)
    if type(copy) is str:
        copy = eval(copy)
    return r.run(array=array,
                 label_index=label_index,
                 with_centering=with_centering,
                 with_scaling=with_scaling,
                 quantile_range=quantile_range,
                 copy=copy)


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