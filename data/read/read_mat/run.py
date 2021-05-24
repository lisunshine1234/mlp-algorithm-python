import scipy.io as sio
def run(file_name, key):
    return {"array": sio.loadmat(file_name)[key].tolist()}

