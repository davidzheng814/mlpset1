import numpy as np
from numpy.linalg import inv
from loadFittingDataP2 import getData
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import Lasso

def plot(X, Y, reg_func):
    dx = 0.01
    x_cont = np.arange(min(X), max(X) + dx, dx)
    y_pred = [reg_func(x) for x in x_cont]
    y_original = [math.cos(math.pi * x) + 1.5 * math.cos(2 * math.pi * x) for x in x_cont]

    plt.plot(X, Y, 'o')
    plt.plot(x_cont, y_pred)
    plt.plot(x_cont, y_original)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def regression(X, Y, basis_funcs, lam=0, is_lasso=False):
    apply_X = []
    for func in basis_funcs:
        apply_X.append(np.apply_along_axis(func, 0, X))

    apply_X_trans = np.array(apply_X)
    apply_X = np.transpose(apply_X_trans)
    m = len(basis_funcs)

    if is_lasso:
        lasso = Lasso(alpha=lam, fit_intercept=False, max_iter=100000)
        lasso.fit(apply_X, Y)
        w = lasso.coef_
    else:
        w = (inv(apply_X_trans.dot(apply_X) + lam * np.identity(m)).dot(apply_X_trans)).dot(Y)

    return w, apply_X

def squared_error(X, Y, w):
    squared_error = sum((X.dot(w) - Y) ** 2)
    gradient = 2 * (X.dot(w) - Y).dot(X)

    return squared_error, gradient

def get_reg_func(w, basis_funcs):
    return lambda x: sum([w_i * func(x) for w_i, func in zip(w, basis_funcs)])

def poly_regression(X, Y, M, lam=0, is_lasso=False):
    def power(i):
        return lambda a: a ** i

    basis_funcs = [power(i) for i in range(M + 1)]
    w, apply_X = regression(X, Y, basis_funcs, lam, is_lasso)
    reg_func = get_reg_func(w, basis_funcs)

    return w, apply_X, reg_func

def cos_regression(X, Y, M, lam=0, is_lasso=False):
    def cos(i):
        return lambda a: np.cos(math.pi * i * a)

    basis_funcs = [cos(i) for i in range(1, M + 1)]
    w, apply_X = regression(X, Y, basis_funcs, lam, is_lasso)
    reg_func = get_reg_func(w, basis_funcs)

    return w, apply_X, reg_func

def special_sin_regression(X, Y, M, lam=0, is_lasso=False):
    def sin(i):
        if i == 0:
            return lambda a: a
        else:
            return lambda a: np.sin(0.4 * math.pi * i * a)

    basis_funcs = [sin(i) for i in range(M)]
    w, apply_X = regression(X, Y, basis_funcs, lam, is_lasso)
    reg_func = get_reg_func(w, basis_funcs)

    return w, apply_X, reg_func

if __name__ == '__main__':
    X, Y = getData(ifPlotData=False)
    M = 13
    lam = .001
    is_lasso = True
    w, apply_X, reg_func = special_sin_regression(X, Y, M, lam, is_lasso)

    print "Regression Weights (w):", w
    squared_error, gradient = squared_error(apply_X, Y, w)
    print "Squared Error:", squared_error
    print "Gradient of Squared Error at w", gradient

    plot(X, Y, reg_func)