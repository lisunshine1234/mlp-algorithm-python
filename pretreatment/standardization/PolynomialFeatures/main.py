import run as r


# F
def main(array, degree=2, interaction_only=False, include_bias=True, order='C'):
    if type(array) is str:
        array = eval(array)
    if type(degree) is str:
        degree = eval(degree)
    if type(interaction_only) is str:
        interaction_only = eval(interaction_only)
    if type(include_bias) is str:
        include_bias = eval(include_bias)
    return r.run(array=array,
                 degree=degree, interaction_only=interaction_only, include_bias=include_bias, order=order)


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
