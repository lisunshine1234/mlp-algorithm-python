from scipy.io import loadmat
import numpy as np
import xlrd as x
import pandas as pd


def run(file, delimiter):
    file_name = file["file"]

    file_type_list = file_name.split(".")
    file_type = file_type_list[len(file_type_list) - 1]

    if file_type == 'mat':
        key = file["key"]
        array = read_mat(file_name, key)
    elif file_type == 'csv':
        array = read_csv(file_name, delimiter)
    elif file_type == 'txt':
        array = read_csv(file_name, delimiter)
    elif file_type == 'xlsx':
        array = read_xls(file_name)
    elif file_type == 'xls':
        array = read_xls(file_name)
    else:
        array = np.array([[]])

    return {"array": array.tolist()}


def read_mat(file, matKey):
    dict = loadmat(file)
    return dict[matKey]


def read_csv(file, delimiter):
    array = np.loadtxt(file, delimiter=delimiter)
    return np.array(array, dtype=float)


def read_xls(file):
    array = pd.read_excel(file, sheet_name=0, header=None)
    return np.array(array, dtype=float)
