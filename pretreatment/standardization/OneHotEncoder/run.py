import numpy as np
from sklearn.preprocessing import OneHotEncoder


def run(array, categories, drop, sparse, dtype, handle_unknown):
    array = np.array(array)
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

    if drop is not None and type(drop) is str and drop != 'first' and drop != 'if_binary':
        drop = eval(drop)

    if type(categories) is str and categories != 'auto':
        if categories.__contains__('[') or categories.__contains__('(') or categories.__contains__(','):
            categories = eval(categories)

    scaler = OneHotEncoder(
        categories=categories,
        drop=drop,
        sparse=sparse,
        dtype=dtype,
        handle_unknown=handle_unknown)
    array = scaler.fit_transform(array).toarray()
    categories_ = []
    for i in scaler.categories_:
        categories_.append(i.tolist())
    drop_idx_ = None
    if scaler.drop_idx_ is not None:
        drop_idx_ = scaler.drop_idx_.tolist()
    return {"array": array.tolist(), 'categories_': categories_, 'drop_idx_': drop_idx_}
