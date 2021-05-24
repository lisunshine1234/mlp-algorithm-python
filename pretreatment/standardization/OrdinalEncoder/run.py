import numpy as np
from sklearn.preprocessing import OrdinalEncoder


def run(array, label_index, categories, dtype):
    x = np.array(array)
    if dtype == 'float':
        dtype = np.float
    elif dtype == 'int':
        dtype = np.int
    elif dtype == 'object':
        dtype = np.object
    elif dtype == 'bool':
        dtype = np.bool
    elif dtype == 'str':
        dtype = np.str

    if type(categories) is str and categories != 'auto':
        if categories.__contains__('[') or categories.__contains__('(') or categories.__contains__(','):
            categories = eval(categories)

    if label_index is not None:
        if label_index < 0:
            label_index = len(array[0]) + label_index

        y = x[:, label_index]
        x = np.delete(array, label_index, axis=1)

    scaler = OrdinalEncoder(categories=categories, dtype=dtype)
    x = scaler.fit(x).transform(x)
    if label_index is not None:
        x = np.insert(x, label_index, values=y, axis=1)
    categories_ = []
    for i in scaler.categories_:
        categories_.append(i.tolist())
    return {"array": x.tolist(), 'categories_': categories_}
