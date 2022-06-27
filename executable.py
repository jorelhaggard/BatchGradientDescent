
import numpy as np
from Activations import sigmoid, regularized_cost
from data import load_data
from BGD import reg_gradient_descent
from Gradient import reg_get_gradient

x_train, y_train, x_test, y_test = load_data('data/ex2data1.txt')
alpha = .0014
iterations = 100000
lambda_ = .01
b_0 = 1
np.random.seed(1)
w_0 = np.random.rand(np.size(x_train[1])) - 0.5
w, b = reg_gradient_descent(x_train, y_train, w_0, b_0, regularized_cost, reg_get_gradient, alpha, iterations, lambda_)
print(w, b)

# GUESSES
predicts = []
error = 0
k = len(x_test)
u = 0
h = 0
for i in range(k):
    z = np.dot(x_test[i], w)+b
    f = sigmoid(z)
    if f>0.5:
        predicts.append(1)
    else:
        predicts.append(0)
    q = abs(predicts[-1]-y_test[i])
    if q == 0:
        u += 1
        h+= 1
    else:
        u += 1
print(f"  {h}  correct out of   {u}  predictions. {h/u*100}%  accuracy.")


