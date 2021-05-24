import numpy as np


def cut(array, row, column):
    array = np.array(array)
    slice_str_check(row)
    slice_str_check(column)
    array = eval('array[' + row + ',' + column + ']')
    return {"array": array.tolist()}


def str_to_int(str):
    try:
        return int(str)
    except ValueError:
        return None


def slice_str_check(str):
    str_split = str.split(":")
    if len(str_split) == 1:
        num = str_to_int(str_split[0])
        if num is None:
            raise Exception('The input "' + str + '" is not an integer!\n' +
                            'The format for example is "1" ":" "1:" ":1" "1:2"')
    elif len(str_split) == 2:
        if len(str_split[0]) > 0:
            start = str_to_int(str_split[0])
            if start is None:
                raise Exception('The input "' + str + '" has a incorrect format!\n' +
                                'The format for example is "1" ":" "1:" ":1" "1:2"')

        if len(str_split[1]) > 0:
            end = str_to_int(str_split[1])
            if end is None:
                raise Exception('The input "' + str + '" has a incorrect format!\n' +
                                'The format for example is "1" ":" "1:" ":1" "1:2"')
    else:
        raise Exception('The input "' + str + '" has a incorrect format!\n' +
                        'The format for example is "1" ":" "1:" ":1" "1:2"')
