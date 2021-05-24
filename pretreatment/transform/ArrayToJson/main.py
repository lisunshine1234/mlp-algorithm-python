def main(*arrays, labels):
    json = {}
    if type(labels) is str:
        labels = labels.split(',')
    if len(arrays) != len(labels):
        raise Exception("标签和数组的长度不相等")
    for i in range(len(labels)):
        json[labels[i]] = arrays[i]
    return {"json": json}


if __name__ == '__main__':
    import numpy as np
    import json

    array = np.loadtxt('D:\\123_2.csv', delimiter=',')
    array = array[0:20, :]

    y = array[:, -1].tolist()
    x = np.delete(array, -1, axis=1).tolist()
    array = array.tolist()
    back = main(x, x, x, labels=[0, 1, 2])

    print(back)
    for i in back:
        print(i + ":" + str(back[i]))

    json.dumps(back)
