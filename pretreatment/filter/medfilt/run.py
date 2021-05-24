import numpy as np
import scipy.signal as signal


def run(array, label_index, label_index_type, label, label_contain, column, column_contain, kernel_size):
    array = np.array(array)
    array_back = None
    column_count = len(array[0])
    label_all = list(set(array[:, label_index]))
    column_all = list(range(column_count))
    column_deal = deal_column(column_all, column_count, column, column_contain, label_index)
    label_list = array[:, label_index]
    array_filter = {}
    for i in label_all:
        array_filter[i] = (label_list == i)

    if label_index_type:
        label_deal = deal_label(label_all, label, label_contain)
        for i in column_all:
            array_temp = array[:, i]
            if column_deal.__contains__(i) > -1:
                for j in label_all:
                    filter_data = array_temp[array_filter[j]]
                    if label_deal.__contains__(j):
                        filter_data = signal_medfilt(filter_data, kernel_size)
                    array_temp[array_filter[j]] = filter_data

            if array_back is None:
                array_back = array_temp.reshape((len(array_temp), 1))
            else:
                array_back = np.hstack((array_back, array_temp.reshape((len(array_temp), 1))))
    else:
        for i in column_all:
            array_temp = array[:, i]
            if column_deal.__contains__(i) > -1:
                array_temp = signal_medfilt(array_temp)

            if array_back is None:
                array_back = array_temp.reshape((len(array_temp), 1))
            else:
                array_back = np.hstack((array_back, array_temp.reshape((len(array_temp), 1))))

    return {"array": array_back.tolist()}


def signal_medfilt(array_one, kernel_size):
    return signal.medfilt(array_one, kernel_size)


def deal_label(label_all, label, label_contain):
    if label_contain is None:
        if label is None:
            label_contain = False
        else:
            label_contain = True
    if label is None:
        label = []
    else:
        label = np.array(label.split(','), dtype=type(label_all[0]))

    if label_contain:
        label_deal = np.intersect1d(label_all, label)
    else:
        label_deal = np.setdiff1d(label_all, label)

    return label_deal


def deal_column(column_all, column_count, column, column_contain, label_index):
    if column_contain is None:
        if column is None:
            column_contain = False
        else:
            column_contain = True

    if column is None:
        column = []
    else:
        column = np.array(column.split(','), dtype=int)

    for i in range(len(column)):
        if column[i] < 0:
            column[i] = column[i] + column_count
    if column_contain:
        column_deal = np.intersect1d(column_all, column)
    else:
        column_deal = np.setdiff1d(column_all, column)

    a = []
    if label_index < 0:
        a.append(column_count + label_index)
    else:
        a.append(label_index)
    column_deal = np.setdiff1d(column_deal, [a])
    return column_deal
