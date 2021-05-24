import numpy as np
from sklearn.preprocessing import PolynomialFeatures


def run(array, degree, interaction_only, include_bias, order):
    array = np.array(array)

    scaler = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias,
                                order=order)
    array = scaler.fit_transform(array)
    return {"array": array.tolist(),
            "n_output_features_": scaler.n_output_features_,
            "n_input_features_": scaler.n_input_features_,
            "powers_": scaler.powers_.tolist()}
