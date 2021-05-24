import run as r


def main(array,
         label_index=None,
         n_quantiles=1000,
         output_distribution='uniform',
         ignore_implicit_zeros=False,
         subsample=int(1e5),
         random_state=None,
         copy=True):
    if type(array) is str:
        array = eval(array)
    if type(label_index) is str:
        label_index = eval(label_index)
    if type(n_quantiles) is str:
        n_quantiles = eval(n_quantiles)
    if type(ignore_implicit_zeros) is str:
        ignore_implicit_zeros = eval(ignore_implicit_zeros)
    if type(subsample) is str:
        subsample = eval(subsample)
    if type(label_index) is str:
        label_index = eval(label_index)
    if type(random_state) is str:
        random_state = eval(random_state)
    if type(copy) is str:
        copy = eval(copy)
    return r.run(array=array,
                 label_index=label_index,
                 n_quantiles=n_quantiles,
                 output_distribution=output_distribution,
                 ignore_implicit_zeros=ignore_implicit_zeros,
                 subsample=subsample,
                 random_state=random_state,
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