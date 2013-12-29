# -*- coding: utf-8 -*-
import theano
import numpy as np
from scipy import linalg


def initialize_weights(fin, fout):
    return np.sin(np.arange(1, fout * (fin+1)+1)).reshape((fin+1, fout)) / 10


def compute_numerical_gradient(J, theta):
    numgrad = np.zeros(len(theta))
    perturb = np.zeros(len(theta))
    e = 1e-6

    for i in range(len(theta)):
        perturb[i] = e
        loss1 = J(theta - perturb)
        loss2 = J(theta + perturb)
        numgrad[i] = (loss2-loss1) / (2*e)
        perturb[i] = 0

    return numgrad


def check_classifier_grad(input_size, output_labels, m, model, x0):

    X = theano.shared(initialize_weights(m-1, input_size))
    y = theano.shared(np.mod(np.arange(0, m), output_labels))

    J, grad = model._cost_and_gradient(X, y)

    J = theano.function([], J)
    grad = theano.function([], grad)

    def f(theta):
        model.theta = theta
        return J()

    def g(theta):
        model.theta = theta
        return grad()

    theta = x0
    print(f(theta))
    gradient = g(theta)
    numgrad = compute_numerical_gradient(f, theta)

    print(np.vstack([gradient, numgrad]).T)
    print(linalg.norm(gradient-numgrad) / linalg.norm(gradient+numgrad))

    theta = theta - 0.01*gradient
    gradient = g(theta)
    numgrad = compute_numerical_gradient(f, theta)
    print(np.vstack([gradient, numgrad]).T)
    print(linalg.norm(gradient-numgrad) / linalg.norm(gradient+numgrad))
