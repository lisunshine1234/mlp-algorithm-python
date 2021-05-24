import run as r


def main(array, label_index=None, threshold=0.0, copy=True):
    if type(array) is str:
        array = eval(array)
    if type(label_index) is str:
        label_index = eval(label_index)
    if type(threshold) is str:
        threshold = eval(threshold)
    if type(copy) is str:
        copy = eval(copy)
    return r.run(array=array,
                 label_index=label_index,
                 copy=copy,
                 threshold=threshold)



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
