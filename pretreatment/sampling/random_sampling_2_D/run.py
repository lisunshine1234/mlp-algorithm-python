import numpy as np


def run(array, row_column, size, replace, p):
    array = np.array(array)
    if row_column:
        sample_index = range(len(array))
    else:
        sample_index = range(len(array[0]))

    sample = np.random.choice(sample_index, size, replace, p)
    if len(sample.shape) == 0:
        sample = [sample]
    if row_column:
        sample_array = []
        for i in sample:
            sample_array.append(array[i])
        sample_array = np.array(sample_array)
    else:
        sample_array = None
        for i in sample:
            if sample_array is None:
                sample_array = array[:, i].reshape((len(array[:, i]), 1))
            else:
                sample_array = np.hstack((sample_array, array[:, i].reshape((len(array[:, i]), 1))))

    return {"sample": sample_array.tolist()}
