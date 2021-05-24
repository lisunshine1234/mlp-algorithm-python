import numpy as np
from sklearn.preprocessing import LabelBinarizer


def run(array, neg_label, pos_label, sparse_output):
    x = np.array(array)

    scaler = LabelBinarizer(neg_label=neg_label, pos_label=pos_label, sparse_output=sparse_output)
    x = scaler.fit_transform(x)
    return {"array": x.tolist(),
            'sparse_input_': scaler.sparse_input_,
            'classes_': scaler.classes_,
            'y_type_': scaler.y_type_}
