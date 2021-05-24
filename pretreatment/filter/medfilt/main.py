import numpy as np
import run as r


def main(array,
         label_index,
         label_index_type,
         label=None,
         label_contain=None,
         column=None,
         column_contain=None,
         kernel_size=5):
    if type(array) is str:
        array = eval(array)
    if type(label_index) is str:
        label_index = eval(label_index)
    if type(label_index_type) is str:
        label_index_type = eval(label_index_type)
    if type(label_contain) is str:
        label_contain = eval(label_contain)
    if type(column_contain) is str:
        column_contain = eval(column_contain)
    if type(kernel_size) is str:
        column_contain = eval(column_contain)

    return r.run(array=array,
                 label_index=label_index,
                 label_index_type=label_index_type,
                 label=label,
                 label_contain=label_contain,
                 column=column,
                 column_contain=column_contain,
                 kernel_size=kernel_size)


if __name__ == '__main__':
    import numpy as np
    import json
    array = np.loadtxt('D:\\123_2.csv', delimiter=',')
    array = array[0:20, :]

    y = array[:, -1].tolist()
    x = np.delete(array, -1, axis=1).tolist()
    array = array.tolist()
    back = main(array, -1, True, '3', False)


    print(back)
    for i in back:
        print(i + ":" + str(back[i]))
    import json

    json.dumps(back)
