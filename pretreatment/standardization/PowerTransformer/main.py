import run as r


# 'box-cox'
def main(array, label_index=None, method='yeo-johnson', standardize=True, copy=True):
    if type(array) is str:
        array = eval(array)
    if type(label_index) is str:
        label_index = eval(label_index)
    if type(standardize) is str:
        standardize = eval(standardize)
    if type(copy) is str:
        copy = eval(copy)
    return r.run(array=array,
                 label_index=label_index,
                 method=method,
                 standardize=standardize,
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