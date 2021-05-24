import numpy as np
import run as r


#     categories : 'auto' or a list of array-like, default='auto'
#     drop : {'first', 'if_binary'} or a array-like of shape (n_features,), default=None
#     sparse : bool, default=True
#     dtype : number type, default=np.float
#     handle_unknown : {'error', 'ignore'}, default='error'
def main(array, categories='auto', drop=None, sparse=True, dtype='float64', handle_unknown='error'):
    if type(array) is str:
        array = eval(array)
    if type(categories) is str and categories != 'auto':
        categories = eval(categories)
    if type(drop) is str and drop != 'first' and drop != 'if_binary':
        drop = eval(drop)
    if type(sparse) is str:
        sparse = eval(sparse)

    return r.run(array=array,
                 categories=categories,
                 drop=drop,
                 sparse=sparse,
                 dtype=dtype,
                 handle_unknown=handle_unknown)


if __name__ == '__main__':
    import numpy as np
    import json

    array = np.loadtxt('D:\\123_2.csv', delimiter=',')
    array = array[0:20, :]

    y = array[:, -1].tolist()
    x = np.delete(array, -1, axis=1).tolist()
    array = array.tolist()
    back = main([y])

    print(back)
    for i in back:
        print(i + ":" + str(back[i]))

    json.dumps(back)
