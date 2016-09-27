import numpy as np
from numpy.linalg import inv
from lassoData import *
import matplotlib
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import Lasso

def plot(train_X, train_Y, valid_X, valid_Y, test_X, test_Y, reg_func, reg_func2, reg_func3):
    dx = 0.01
    train_X = list(train_X)
    train_Y = list(train_Y)
    valid_X = list(valid_X)
    valid_Y = list(valid_Y)
    test_X = list(test_X)
    test_Y = list(test_Y)
    x_cont = np.arange(min(valid_X + train_X + test_X), max(valid_X + train_X + test_X) + dx, dx)
    y_pred1 = [reg_func(x) for x in x_cont]
    y_pred2 = [reg_func2(x) for x in x_cont]
    y_pred3 = [reg_func3(x) for x in x_cont]
    y_original = [5.646300000000000100e+00 * np.sin(0.4 * np.pi * x * 2)
                  + 7.785999999999999600e-01 * np.sin(0.4 * np.pi * x * 3)
                  + 8.108999999999999500e-01 * np.sin(0.4 * np.pi * x * 5)
                  + 2.682700000000000100e+00 * np.sin(0.4 * np.pi * x * 6)
                  for x in x_cont]

    fig = plt.figure(1)
    fig.patch.set_facecolor('white')
    fig.set_figheight(5)
    fig.set_figwidth(5) 

    ax = fig.add_subplot(1, 1, 1)

    plt.plot(train_X, train_Y, 'o')
    plt.plot(valid_X, valid_Y, 'o')
    plt.plot(test_X, test_Y, 'o')

    plt.plot(x_cont, y_pred1)
    plt.plot(x_cont, y_pred2)
    plt.plot(x_cont, y_pred3)
    plt.plot(x_cont, y_original)
    plt.title(u'Linear Regression (M = 10, \u03BB = 0.1)')
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

    if is_lasso and lam != 0.:
        lasso = Lasso(alpha=lam, fit_intercept=False, max_iter=10000000)
        lasso.fit(apply_X, Y)
        w = lasso.coef_
    else:
        w = (inv(apply_X_trans.dot(apply_X) + lam * np.identity(m)).dot(apply_X_trans)).dot(Y)

    return w, apply_X

def squared_error(X, Y, w):
    squared_error = np.mean((X.dot(w) - Y) ** 2)
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
    print w
    reg_func = get_reg_func(w, basis_funcs)

    return w, apply_X, reg_func

if __name__ == '__main__':
    X, Y = lassoTrainData()
    X2, Y2 = lassoValData()
    X3, Y3 = lassoTestData()

    w, apply_X, reg_func = special_sin_regression(X, Y, 13, 0, False)
    w, apply_X, reg_func2 = special_sin_regression(X, Y, 13, 1e-5, False)
    w2, apply_X2, reg_func3 = special_sin_regression(X, Y, 13, 1e-5, True)

    plot(X, Y, X2, Y2, X3, Y3, reg_func, reg_func2, reg_func3)