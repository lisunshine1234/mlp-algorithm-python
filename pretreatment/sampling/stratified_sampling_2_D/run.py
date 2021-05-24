import numpy as np


def run(array, label_index, label_dict,  replace):
    array = np.array(array)
    label_all = list(set(array[:, label_index]))
    if label_dict is None:
        label_dict = {}
    samples = []
    for label in label_all:
        if label_dict.keys().__contains__(label):
            size = label_dict[label]
        else:
            size = 0.5
        rows = array[array[:, label_index] == label]
        row_len = len(rows)

        if size < 1:
            size = size * row_len
        size = int(size)
        sample = np.random.choice(range(row_len), size, replace)
        for i in sample:
            samples.append(rows[i].tolist())

    return {"sample": samples}
