import run as r


def main(array, n_bins=5, encode='onehot', strategy='quantile'):
    if type(array) is str:
        array = eval(array)
    if type(n_bins) is str:
        n_bins = eval(n_bins)
    return r.run(array=array,
                 n_bins=n_bins,
                 encode=encode,
                 strategy=strategy)



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
