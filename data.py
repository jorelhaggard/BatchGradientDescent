import numpy as np

def load_data(filename):
    data = np.loadtxt(filename, delimiter=',')
    X = data[:79,:2]
    y = data[:79,2]
    xv = data[79:,:2]
    yv = data[79:,2]
    return X, y, xv, yv
