from scipy.io import loadmat
import numpy as np


def run(fileName):
    file = fileName["file"]
    if "key" in fileName:
        mat = fileName["key"]

    file_type_list = file.split(".")
    file_type = file_type_list[len(file_type_list) - 1]

    if file_type == 'mat':
        back = read_mat(file, mat)
    elif file_type == 'csv':
        back = read_csv(file)
    elif file_type == 'txt':
        back = read_txt(file)
    else:
        back = np.array([[]])

    return {"set": back.tolist()}


def read_mat(file, matKey):
    m = loadmat(file)
    return m[matKey]


def read_csv(file):
    set = np.loadtxt(file, delimiter=',')
    return np.array(set, dtype=float)


def read_txt(file):
    data = np.loadtxt(file, delimiter=',')
    return np.array(data, dtype=float)