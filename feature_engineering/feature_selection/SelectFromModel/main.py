import numpy as np
import run as r


def main(x, y, estimator, threshold=None, prefit=False, norm_order=1, max_features=None):

    if type(x) is str:
        x = eval(x)
    if type(y) is str:
        y = eval(y)
    if type(threshold) is str and threshold != 'mean' and threshold != 'median':
        threshold = eval(threshold)
    if type(prefit) is str:
        prefit = eval(prefit)
    if type(norm_order) is str:
        norm_order = eval(norm_order)
    if type(max_features) is str:
        max_features = eval(max_features)
    return r.run(x=x, y=y,
                 estimator=estimator,
                 threshold=threshold,
                 prefit=prefit,
                 norm_order=norm_order,
                 max_features=max_features)


if __name__ == '__main__':
    import numpy as np
    import json

    array = np.loadtxt('D:\\123_2.csv', delimiter=',')
    array = array[0:20, :]

    y = array[:, -1].tolist()
    x = np.delete(array, -1, axis=1).tolist()
    array = array.tolist()
    back = main(x, y,'SVC')

    print(back)
    for i in back:
        print(i + ":" + str(back[i]))

    json.dumps(back)