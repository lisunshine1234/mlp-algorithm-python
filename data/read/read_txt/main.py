import run as r


def main(file_name, delimiter=','):
    return r.run(file_name=file_name,
                 delimiter=delimiter)


if __name__ == '__main__':
    import numpy as np
    import json
    print(main('D:\\aaa.xlsx', ','))
