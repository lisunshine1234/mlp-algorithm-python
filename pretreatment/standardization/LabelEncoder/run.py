import numpy as np
from sklearn.preprocessing import LabelEncoder


def run(array):
    x = np.array(array)

    scaler = LabelEncoder()
    x = scaler.fit_transform(x)
    return {"array": x.tolist(), "classes_": scaler.classes_.tolist()}
