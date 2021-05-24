import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer


def run(array, sparse_output, classes):
    array = np.array(array)
    scaler = MultiLabelBinarizer(classes=classes, sparse_output=sparse_output)
    array = scaler.fit_transform(array)
    return {"array": array.tolist(), 'classes_': scaler.classes_.tolist()}
