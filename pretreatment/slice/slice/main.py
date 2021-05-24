import run as r


def main(array, row=':', column=':'):
    if type(array) is str:
        array = eval(array)
    return r.cut(array, row, column)


if __name__ == '__main__':
    import numpy as np
    import json
    print(main([[1, 2, 3], [2, 3, 6]], ':', ':'))
