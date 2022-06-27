import numpy as np
from Activations import sigmoid

def reg_get_gradient(x, y, w, b, lambda_):

    m, n = x.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0
    for i in range(m):
        z = 0
        z += np.dot(x[i], w)+b
        f_wb = sigmoid(z)
        err = f_wb-y[i]
        for j in range(n):
            dj_dw_ij = err*x[i][j]
            dj_dw[j] += dj_dw_ij
        dj_db = dj_db + err
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    # REGULARIZATION
    for j in range(n):
        dj_dw_j = w[j]*(lambda_/m)
        dj_dw[j] += dj_dw_j
    return dj_db, dj_dw




