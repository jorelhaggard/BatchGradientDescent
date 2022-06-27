import numpy as np
from Activations import sigmoid, regularized_cost
from data import load_data
from executable import reg_gradient_descent
from Gradient import reg_get_gradient

x_train, y_train, x_test, y_test = load_data('data/ex2data1.txt')
alpha = .01
iterations = 10000
lambda_ = .1
b_0 = 1
w_0 = np.random.rand(x_train.shape)-0.5
w, b = reg_gradient_descent(x_train, y_train, w_0, b_0, regularized_cost, reg_get_gradient, alpha, iterations, lambda_)