import run as r


def main(array, label_index=None, categories='auto', dtype='float'):
    if type(array) is str:
        array = eval(array)
    if type(categories) is str and categories != 'auto':
        categories = eval(categories)
    if type(label_index) is str :
        label_index = eval(label_index)
    return r.run(array=array, label_index=label_index, categories=categories, dtype=dtype)


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