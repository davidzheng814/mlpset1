import numpy as np
from numpy.linalg import inv
from loadFittingDataP2 import getData
import matplotlib.pyplot as plt
import math

def get_reg_func(w, basis_funcs):
    return lambda x: sum([w_i * func(x) for w_i, func in zip(w, basis_funcs)])

def plot(X, Y, reg_func):
    dx = 0.01
    x_cont = np.arange(min(X), max(X), dx)
    y_pred = [reg_func(x) for x in x_cont]
    y_original = [math.cos(math.pi * x) + 1.5 * math.cos(2 * math.pi * x) for x in x_cont]

    plt.plot(X, Y, 'o')
    plt.plot(x_cont, y_pred)
    plt.plot(x_cont, y_original)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def mle(X, Y, basis_funcs):
    data_X = []
    for func in basis_funcs:
        data_X.append(np.apply_along_axis(func, 0, X))

    data_X_trans = np.array(data_X)
    data_X = np.transpose(data_X_trans)

    w = (inv(data_X_trans.dot(data_X)).dot(data_X_trans)).dot(Y)

    return w

def squared_error(X, Y, reg_func, deriv_func=None):
    errors = (np.apply_along_axis(reg_func, 0, X) - Y)
    squared_error = sum(errors ** 2)
    if deriv_func:
        deriv_squared_error = sum(2 * errors * np.apply_along_axis(deriv_func, 0, X))
        return squared_error, deriv_squared_error
    else:
        return squared_error


def poly_mle(X, Y, M):
    def power(i):
        return lambda a: a ** i
    def deriv(i):
        if i == 0:
            return lambda a: 0
        return lambda a: i * a ** (i - 1)

    basis_funcs = [power(i) for i in range(M)]
    deriv_funcs = [deriv(i) for i in range(M)]
    w = mle(X, Y, basis_funcs)

    return w, basis_funcs, deriv_funcs

X, Y = getData(ifPlotData=False)
M = 5
w, basis_funcs, deriv_funcs = poly_mle(X, Y, M)
reg_func = get_reg_func(w, basis_funcs)
deriv_func = get_reg_func(w, deriv_funcs)
# plot(X, Y, reg_func)
print squared_error(X, Y, reg_func, deriv_func)