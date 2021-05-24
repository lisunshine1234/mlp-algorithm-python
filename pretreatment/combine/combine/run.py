import numpy as np


def run(array, combine_type):
    if len(array) == 0:
        return []
    elif len(array) == 1:
        return array[0]
    else:
        back_array = array[0]
        for i in range(1, len(array)):
            if combine_type:
                back_array = np.vstack((back_array, array[i]))
            else:
                back_array = np.hstack((back_array, array[i]))
    return {"array": back_array.tolist()}
