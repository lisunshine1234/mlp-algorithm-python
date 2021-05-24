import run as r

def main(*array, combine_type=True):
    array_ = []
    for i in array:
        if type(i) is str:
            i = eval(i)
        array_.append(i)
    if type(combine_type) is str:
        combine_type = eval(combine_type)
    return r.run(array=array_,
                 combine_type=combine_type)


if __name__ == '__main__':
    import numpy as np
    import json
    array = np.loadtxt('D:\\123_2.csv', delimiter=',')
    array = array[0:20, :]

    y = array[:, -1].tolist()
    x = np.delete(array, -1, axis=1).tolist()

    back = main(x, x, x)
    print(back)
    for i in back:
        print(i + ":" + str(back[i]))
    import json
    json.dumps(back)
