import run as r


def main(fileName, delimiter=','):
    return r.run(fileName, delimiter)


if __name__ == '__main__':
    import numpy as np
    import json
    print(main({'file': 'D:\\aaa.xlsx', 'key': 'Data'}))
