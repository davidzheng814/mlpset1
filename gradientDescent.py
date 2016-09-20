import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp
import loadParametersP1
import loadFittingDataP1
import random

class NegativeGaussian():
    # The mean and covariance parameters must both be n x n arrays
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov
        self.multivariate = sp.multivariate_normal(mean, cov)
    # Returns the scalar value
    def value(self, X):
        return np.negative(self.multivariate.pdf(X))
    # Returns a vector
    def gradient(self, X):
        return self.multivariate.pdf(X) * (np.linalg.inv(self.cov)).dot(X - self.mean)

class QuadraticBowl():
    # A must be a positive definite n x n array, B must be a 1 x n array
    def __init__(self, A, B):
        self.A = A
        self.B = B
    # Returns the scalar value
    def value(self, X):
        return 0.5 * X.T.dot(self.A).dot(X) - X.T.dot(self.B)
    # Returns a vector
    def gradient(self, X):
        return self.A.dot(X) - self.B

# In order to perform gradient descent on a given function, we start at an initialGuess and calculate the gradient.
# Then, we multiply the gradient by stepSize and use that to perturb our initial guess. We continue in this fashion
# until the norm of the gradient is less than the threshold
# function must be a class with a value(X) and gradient(X) method
# initialGuess must be a vector
# stepSize must be a scalar
# threshold must be a scalar
# should only plot when in 2D
def gradientDescent(function, initialGuess, stepSize, threshold, shouldPlot = False):
    guess = initialGuess
    gradient = function.gradient(guess)

    if (shouldPlot):
        xArray = np.array(guess[0])
        yArray = np.array(guess[1])
        normArray = np.array(np.linalg.norm(gradient))

    while (np.linalg.norm(gradient) > threshold):
        guess = guess - stepSize * gradient
        gradient = function.gradient(guess)
        if (shouldPlot):
            xArray = np.append(xArray, guess[0])
            yArray = np.append(yArray, guess[1])
            normArray = np.append(normArray, np.linalg.norm(gradient))
    
    if (shouldPlot):
        plot(xArray, yArray, 'x', 'y', 'Guesses in Gradient Descent in the 2D Plane')
        plot(np.arange(normArray.size), normArray, 'Trial Number', 'Gradient Norm', 'Gradient Norm vs. Trial Number')

    return guess

# Plots the data given in X and Y
# X and Y must be vectors of the same size
def plot(X, Y, xlabel, ylabel, title):
    plt.plot(X,Y,'o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

# Numerically calculates the gradient of the function at point X with the central difference approximation
# Uses the stepSize to go a little bit above and below to calculate each part of the gradient
# X must be a vector
# stepSize must be a scalar
# must be in 2 dimensions (or else I'll have to generalize this and I don't want to)
def numericalGradient(function, X, stepSize):
    xGuess = (function.value(X + np.array([0.5 * stepSize, 0])) - function.value(X - np.array([0.5 * stepSize, 0]))) / stepSize
    yGuess = (function.value(X + np.array([0, 0.5 * stepSize])) - function.value(X - np.array([0, 0.5 * stepSize]))) / stepSize
    guess = np.array([xGuess, yGuess])
    actual = function.gradient(X)
    print "Guess is " + np.array_str(guess) + ", actual is " + np.array_str(actual)

class SumSquaredErrors():
    # Each row of X and y represents a single data sample
    # X is a n x d array, y is a vector of length n
    def __init__(self, X, y):
        self.X = X
        self.y = y
    # Returns the scalar value of the norm of the sum squared errors
    def value(self, theta):
        a = self.X.dot(theta) - self.y
        return a.T.dot(a)
    # Returns a vector
    def gradient(self, theta):
        return 2 * self.X.T.dot( self.X.dot(theta) - self.y)

def stochasticGradientDescent(X, y, initialGuess, threshold):
    t = 1 # trial number
    n = y.size
    order = range(n) # what order to take data points in
    guess = initialGuess
    pastGuess = initialGuess

    # stop when your current guess is almost the same as the one a whole cycle ago
    while (t == 1 or np.linalg.norm(guess - pastGuess) > threshold):
        random.shuffle(order)
        pastGuess = guess
        for i in order:
            xPoint = X[i] # vector of x values for single data point
            yPoint = y[i] # y value for single data point
            guess = guess - ((t + 1000) ** -.999) * (2 * xPoint.T.dot(xPoint.dot(guess) - yPoint))
            t += 1
    return guess


# ----------------------------- ACTUAL TESTING CODE HERE ---------------------------------

if __name__ == '__main__':
    gaussMean,gaussCov,quadBowlA,quadBowlB = loadParametersP1.getData()

    gauss = NegativeGaussian(gaussMean, gaussCov)
    quad = QuadraticBowl(quadBowlA, quadBowlB)

    # NUMBER 1
    # gradientDescent(gauss, np.array([9.9,9.9]), 100, .00001, True)
    # gradientDescent(quad, np.array([0,0]), .01, .00001, True)

    # NUMBER 2
    # numericalGradient(gauss, np.array([5,0]), .1)

    # NUMBER 3
    X, y = loadFittingDataP1.getData()
    sse = SumSquaredErrors(X, y)
    print gradientDescent(sse, np.arange(10), .000001, .001, False)
    print stochasticGradientDescent(X, y, np.arange(10), .001)
    print np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)