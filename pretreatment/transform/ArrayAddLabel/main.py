import run as r


def main(array, label_x=None, label_y=None):
    if type(array) is str:
        array = eval(array)
    return r.run(array=array,
                 label_x=label_x,
                 label_y=label_y)


if __name__ == '__main__':
    import numpy as np
    import json

    array = np.loadtxt('D:\\123_2.csv', delimiter=',')
    array = array[0:20, :]

    y = array[:, -1].tolist()
    x = np.delete(array, -1, axis=1).tolist()

    back = main(x, label_y=None, label_x="A,B,C,D,E,F,H")
    print(back)
    for i in back:
        print(i + ":" + str(back[i]))
    import json

    json.dumps(back)
