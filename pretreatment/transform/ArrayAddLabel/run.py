import numpy as np



def run(array, label_x, label_y):
    back = {}

    back_array = array

    back['data'] = back_array
    if type(back['data'][0]) is list:
        if label_x is not None:
            label_x = label_x.split(',')
            if len(label_x) != len(back_array[0]):
                raise Exception("需要拼接x轴长度和标签的长度不一致")
        else:
            label_x = list(range(len(back_array[0])))

        if label_y is not None:
            label_y = label_y.split(',')
            if len(label_y) != len(back_array):
                raise Exception("需要拼接y轴方向和标签的长度不一致")
        else:
            label_y = list(range(len(back_array)))
        back['x'] = label_x
        back['y'] = label_y
    else:
        if label_x is not None:
            label_x = label_x.split(',')
            if len(label_x) != len(back_array):
                raise Exception("需要拼接x轴长度和标签的长度不一致")
        else:
            label_x = list(range(len(back_array)))
        back['x'] = label_x
    return {"array": back}
