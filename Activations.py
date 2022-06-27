import numpy as np


def sigmoid(x):
    g = 1.0 / (1.0+np.exp(-x))

    return g

def relu(x):
    return np.maximum(0, x)

def regularized_cost(x, y, w, b, lambda_):
    m, n = x.shape
    loss_sum = 0
    for i in range(m):
        z_wb_i = np.dot(x[i],w)+b
        f_wb_i = sigmoid(z_wb_i)
        if y[i] == 1.:
            loss = -1*np.log(f_wb_i+.0000000000000000000000000001)
        else:
            loss = -1*np.log(1-f_wb_i+.000000000000000000000000000001)
        loss_sum += loss
    total_cost = loss_sum/m
    # REGULARIZATION
    reg_cost = 0
    for j in range(n):
        reg_cost += w[j]**2
    total_cost += (lambda_/(2 * m))*reg_cost
    return total_cost

