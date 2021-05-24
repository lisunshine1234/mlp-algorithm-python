import run as r


def main(array, neg_label=0, pos_label=1, sparse_output=False):
    if type(array) is str:
        array = eval(array)
    if type(neg_label) is str:
        neg_label = eval(neg_label)
    if type(pos_label) is str:
        pos_label = eval(pos_label)
    if type(sparse_output) is str:
        sparse_output = eval(sparse_output)

    return r.run(array=array,
                 neg_label=neg_label,
                 pos_label=pos_label,
                 sparse_output=sparse_output)


if __name__ == '__main__':
    import numpy as np
    import json
    print(main([7, 2, 3]))

