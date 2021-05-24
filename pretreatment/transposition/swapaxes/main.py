import run as r


def main(array, axis1, axis2):
    if type(array) is str:
        array = eval(array)
    if type(axis1) is str:
        axis1 = eval(axis1)
    if type(axis2) is str:
        axis2 = eval(axis2)
    return r.run(array, axis1, axis2)

if __name__ == '__main__':
    import numpy as np
    import json

    array = np.loadtxt('D:\\123_2.csv', delimiter=',')
    array = array[0:20, :]

    y = array[:, -1].tolist()
    x = np.delete(array, -1, axis=1).tolist()
    array = array.tolist()
    back = main(array,1,1)

    print(back)
    for i in back:
        print(i + ":" + str(back[i]))

    json.dumps(back)