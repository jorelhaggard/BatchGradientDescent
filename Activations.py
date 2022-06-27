import numpy as np

def sigmoid(x):
    g = 1.0 / 1.0+np.exp(-x)

    return g

def relu(x):
    return np.maximum(0, x)

def regularized_cost(x, y, w, b, lambda_):
    m, n = x.shape
    loss_sum = 0
    for i in range(m):
        z_wb = np.dot(x[i],w)+b
        f_wb = sigmoid(z_wb)
        loss = -y[i]*np.log(f_wb)-(1-y[i])*np.log(1-f_wb)
        loss_sum += loss
    total_cost = loss_sum/m
    # REGULARIZATION
    reg_cost = 0
    for j in range(n):
        reg_cost += w[j]**2
    total_cost = total_cost+(lambda_/(2 * m))*reg_cost
    return total_cost

