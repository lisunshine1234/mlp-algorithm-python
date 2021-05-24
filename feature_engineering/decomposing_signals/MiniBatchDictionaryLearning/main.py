import run as r


def main(array, label_index=None, n_components=None, copy=True, whiten=False, svd_solver='auto', tol=0.0,
         iterated_power='auto', random_state=None):
    return r.run(array=array,
                 label_index=label_index,
                 n_components=n_components,
                 copy=copy,
                 whiten=whiten,
                 svd_solver=svd_solver,
                 tol=tol,
                 iterated_power=iterated_power,
                 random_state=random_state)


if __name__ == '__main__':
    import numpy as np
    import json
    array = np.loadtxt('D:\\123_2.csv', delimiter=',')
    y = array[:, -1]
    x = np.delete(array, -1, axis=1)

    x = x.tolist()
    y = y.tolist()
    array = array.tolist()
    print(main(array,-1,n_components=2))
