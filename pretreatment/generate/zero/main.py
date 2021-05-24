import numpy as np


def main(row, column):
    return {"array": np.zeros((row, column)).tolist()}


if __name__ == '__main__':
    import json

    array = np.loadtxt('D:\\123_2.csv', delimiter=',')
    array = array[0:20, :]

    y = array[:, -1].tolist()
    x = np.delete(array, -1, axis=1).tolist()
    array = array.tolist()
    back = main(3, 2)

    print(back)
    for i in back:
        print(i + ":" + str(back[i]))

    json.dumps(back)
