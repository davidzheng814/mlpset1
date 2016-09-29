import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp
import loadParametersP1
import loadFittingDataP1
import random
import loadFittingDataP2
from regression import poly_regression

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
    numTrials = 0

    if (shouldPlot):
        xArray = np.array(guess[0])
        yArray = np.array(guess[1])
        normArray = np.array(np.linalg.norm(gradient) * 1000000)

    while (np.linalg.norm(gradient) > threshold):
        guess = guess - stepSize * gradient
        gradient = function.gradient(guess)
        numTrials += 1
        if numTrials % 1000000 == 0:
            print guess
            print np.linalg.norm(gradient)
            print function.value(guess)
        if (shouldPlot):
            xArray = np.append(xArray, guess[0])
            yArray = np.append(yArray, guess[1])
            normArray = np.append(normArray, np.linalg.norm(gradient) * 1000000)
    
    if (shouldPlot):
        # plot(xArray, yArray, 'x', 'y', 'Guesses in Gradient Descent in the 2D Plane')
        plot(np.arange(normArray.size), normArray, 'Trial Number', 'Gradient Norm (in E-6)', 'Gradient Norm vs. Trial for Gaussian')
    
    return (guess, numTrials)

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
    def getX(self):
        return self.X
    def getY(self):
        return self.y

def stochasticGradientDescent(sse, initialGuess, threshold, stepSize):
    t = 0 # trial number
    X = sse.getX()
    y = sse.getY()
    n = y.size
    batch_size = n * 100 # the number of iterations we do before we check for the stopping condition
    guess = initialGuess # our guess for theta
    pastObjectives = [0] * 10 # the last ten objective functions

    # stop when your the objective function is almost the same as the one a cycle ago
    while (t == 0 or max(pastObjectives) - min(pastObjectives) > threshold):
    # while (t == 0 or np.linalg.norm(gradient) > threshold):
        pastObjectives.append(sse.value(guess))
        pastObjectives.pop(0)
        index = random.randint(0, n - 1)
        xPoint = X[index] # vector of x values for single data point
        yPoint = y[index] # y value for single data point
        guess = guess - stepSize(t) * (2 * xPoint.T.dot(xPoint.dot(guess) - yPoint))
        t += 1
        # gradient = sse.gradient(guess)
        if t % (batch_size * 10) == 0:
            # print np.linalg.norm(gradient)
            print guess
            print max(pastObjectives), min(pastObjectives), max(pastObjectives) - min(pastObjectives)
            print t
    return guess, t

## Plot formatting

font = {'family': 'serif', 'weight': 'normal', 'size':20}
plt.rc('font', **font)


# ----------------------------- ACTUAL TESTING CODE HERE ---------------------------------

def error(calculated, actual):
    return np.linalg.norm(calculated - actual) / np.linalg.norm(actual)

def stochasticGradientDescentWithPlot(sse, initialGuess, threshold, stepSize): #for 1-3
    t = 0 # trial number
    X = sse.getX()
    y = sse.getY()
    n = y.size
    guess = initialGuess # our guess for theta
    pastObjectives = [0] * 10 # the last ten objective functions

    normArray = np.array([np.linalg.norm(sse.gradient(guess)) / 1000000])
    valueArray = np.array([sse.value(guess) / 1000000])

    # stop when your the objective function is almost the same as the one a cycle ago
    while (t == 0 or max(pastObjectives) - min(pastObjectives) > threshold):
        pastObjectives.append(sse.value(guess))
        pastObjectives.pop(0)
        index = random.randint(0, n - 1)
        xPoint = X[index] # vector of x values for single data point
        yPoint = y[index] # y value for single data point
        guess = guess - stepSize(t) * (2 * xPoint.T.dot(xPoint.dot(guess) - yPoint))
        t += 1

        normArray = np.append(normArray, np.linalg.norm(sse.gradient(guess)) / 1000000)
        valueArray = np.append(valueArray, sse.value(guess)/ 1000000)

    # plot(np.arange(normArray.size), normArray, 'Trial Number', 'Gradient Norm (in millions)', 'Gradient Norm vs. Trial Number')
    plot(np.arange(valueArray.size), valueArray, 'Trial Number', 'Objective Value Function (in millions)', 'Objective Value vs. Trial Number')

    return guess, t

if __name__ == '__main__':
    gaussMean,gaussCov,quadBowlA,quadBowlB = loadParametersP1.getData()

    gauss = NegativeGaussian(gaussMean, gaussCov)
    quad = QuadraticBowl(quadBowlA, quadBowlB)
    quadSolution = np.array([80./3, 80./3])

    # NUMBER 1
    # initialGuesses = [np.array([9.9, 9.9]), np.array([9,9]), np.array([0,0]), np.array([-90, -90]), np.array([-990, -990])]
    # for initial in initialGuesses:
    #     guess, numTrials = gradientDescent(gauss, initial, 1e6, 1e-9, False)
    #     print "Gave " + str(error(guess, gaussMean)) + " error with " + str(numTrials) + " trials with initial " + str(initial)

    # print ""

    # stepSizes = [1e4, 1e5, 1e6, 1e7, 1e8]
    # for step in stepSizes:
    #     guess, numTrials = gradientDescent(gauss, np.array([0, 0]), step, 1e-9, False)
    #     print "Gave " + str(error(guess, gaussMean)) + " error with " + str(numTrials) + " trials with step size " + str(step)

    # print ""

    # thresholds = [1e-7, 1e-8, 1e-9, 1e-10, 1e-11]
    # for threshold in thresholds:
    #     guess, numTrials = gradientDescent(gauss, np.array([0, 0]), 1e6, threshold, False)
    #     print "Gave " + str(error(guess, gaussMean)) + " error with " + str(numTrials) + " trials with threshold " + str(threshold)

    # initialGuesses = [np.array([26.5, 26.5]), np.array([23,23]), np.array([0,0]), np.array([-240, -240]), np.array([-2600, -2600])]
    # for initial in initialGuesses:
    #     guess, numTrials = gradientDescent(quad, initial, 1e-2, 1e-2, False)
    #     print "Gave " + str(error(guess, quadSolution)) + " error with " + str(numTrials) + " trials with initial " + str(initial)

    # print ""

    # stepSizes = [1, 1e-1, 1e-2, 1e-3, 1e-4]
    # for step in stepSizes:
    #     guess, numTrials = gradientDescent(quad, np.array([0, 0]), step, 1e-2, False)
    #     print "Gave " + str(error(guess, quadSolution)) + " error with " + str(numTrials) + " trials with step size " + str(step)

    # print ""

    # thresholds = [1, 1e-1, 1e-2, 1e-3, 1e-4]
    # for threshold in thresholds:
    #     guess, numTrials = gradientDescent(quad, np.array([0, 0]), 1e-2, threshold, False)
    #     print "Gave " + str(error(guess, quadSolution)) + " error with " + str(numTrials) + " trials with threshold " + str(threshold)
    
    # print gradientDescent(gauss, np.array([0,0]), 1e6, 1e-9, True)
    # print gradientDescent(quad, np.array([0,0]), .01, .01, True)

    # NUMBER 2
    # numericalGradient(gauss, np.array([20,20]), 100)
    # numericalGradient(gauss, np.array([20,20]), 10)
    # numericalGradient(gauss, np.array([20,20]), 1)
    # numericalGradient(gauss, np.array([100,50]), 100)
    # numericalGradient(gauss, np.array([100,50]), 10)
    # numericalGradient(gauss, np.array([100,50]), 1)

    # numericalGradient(quad, np.array([20,20]), 100)
    # numericalGradient(quad, np.array([20,20]), 10)
    # numericalGradient(quad, np.array([20,20]), 1)
    # numericalGradient(quad, np.array([100,50]), 100)
    # numericalGradient(quad, np.array([100,50]), 10)
    # numericalGradient(quad, np.array([100,50]), 1)

    # NUMBER 3
    X, y = loadFittingDataP1.getData()
    sse = SumSquaredErrors(X, y)
    stepSize = lambda t: 6e-3 * ((t + 100) ** -.99)
    # w, trials = gradientDescent(sse, np.zeros(10), .00001, 1, False)
    stoch_w, stoch_trials = stochasticGradientDescent(sse, np.zeros(10), 10, stepSize)
    # stoch_w, stoch_trials = stochasticGradientDescentWithPlot(sse, np.zeros(10), 100, stepSize)
    actual_w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    print sse.value(actual_w)
    # print sse.value(actual_w)
    # print "diff is " + str(np.linalg.norm(w - actual_w) / np.linalg.norm(actual_w))
    print "diff is " + str(np.linalg.norm(stoch_w - actual_w) / np.linalg.norm(actual_w))
    # print stoch_w
    # print w
    # print sse.value(w)
    # print trials
    print stoch_w
    print sse.value(stoch_w)
    print stoch_trials

    # SECOND QUESTION
    # X, Y = loadFittingDataP2.getData(ifPlotData=False)
    # M = 5
    # w, apply_X, reg_func = poly_regression(X,Y,M)
    # sse = SumSquaredErrors(apply_X,Y)

    # numerical_w, trials = gradientDescent(sse, np.arange(M+1), .052, 1e-4, False)
    # print "diff is " + str(np.linalg.norm(numerical_w - w) / np.linalg.norm(w))
    # print numerical_w
    # print sse.value(numerical_w)
    # print w
    # print sse.value(w)
    # print trials

    #lambda t: 400 * (t + 1000) ** -.9999
    # initial = np.zeros(M+1)
    # numerical_w, trials = stochasticGradientDescent(sse, initial, .000001, lambda t: 1e4 * (t + 1e6) ** -.999)
    # print "diff is " + str(np.linalg.norm(numerical_w - w) / np.linalg.norm(w))
    # print numerical_w
    # print sse.value(numerical_w)
    # print w
    # print sse.value(w)
    # print trials
    # print w