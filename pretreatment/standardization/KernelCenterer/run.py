import numpy as np
from sklearn.preprocessing import KernelCenterer


def run(array):
    x = np.array(array)

    scaler = KernelCenterer()
    x = scaler.fit_transform(x)
    return {"array": x.tolist(), 'K_fit_all_': scaler.K_fit_all_.tolist(), 'K_fit_rows_': scaler.K_fit_rows_.tolist()}
