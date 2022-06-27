import numpy as np

def reg_gradient_descent(x, y, w_in, b_in, regularized_cost, reg_get_gradient, alpha, num_iters, lambda_):

    m = len(x)
    J_hist = []
    w_hist = []

    for i in range(num_iters):
        dj_db, dj_dw = reg_get_gradient(x, y, w_in, b_in, lambda_)

        w_in = w_in-alpha*dj_dw
        b_in = b_in-alpha*dj_db
        # VISUAL
        if i<100000:
            cost = regularized_cost(x, y, w_in, b_in, lambda_)
            J_hist.append(cost)
        if i % np.ceil(num_iters/10) == 0 or i == (num_iters-1):
            w_hist.append(w_in)
            print(f"Iteration {i:5}: Cost {float(J_hist[-1]):9.3f}  ")
    return w_in, b_in


