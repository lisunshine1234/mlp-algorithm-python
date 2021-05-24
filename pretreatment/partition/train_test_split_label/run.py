from sklearn.model_selection import train_test_split
import numpy as np


def run(array, label_index, test_size, train_size, random_state, shuffle):
    array = np.array(array)
    y = array[:, label_index]
    x = np.delete(array, label_index, axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, train_size=train_size,
                                                        random_state=random_state,
                                                        shuffle=shuffle)
    return {"x_train": x_train.tolist(), "x_test": x_test.tolist(), "y_train": y_train.tolist(), "y_test": y_test.tolist()}
