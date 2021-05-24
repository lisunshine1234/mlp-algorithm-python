import numpy as np
from sklearn.preprocessing import KBinsDiscretizer


def run(array, n_bins, encode, strategy):
    x = np.array(array)

    scaler = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
    x = scaler.fit_transform(x).toarray()
    bin_edges_ = []
    for i in scaler.bin_edges_:
        bin_edges_.append(i.tolist())
    return {"array": x.tolist(), 'bin_edges_': bin_edges_, 'n_bins_': scaler.n_bins_.tolist()}
