#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import theano
import numpy as np
from scipy import linalg
from nylearn.nnet import nnet


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


def check_NNGradients(lmbd):
    input_layer_size = 3
    hidden_layer1_size = 5
    hidden_layer2_size = 4
    num_labels = 3
    m = 10

    theta1 = initialize_weights(input_layer_size, hidden_layer1_size)
    theta2 = initialize_weights(hidden_layer1_size, hidden_layer2_size)
    theta3 = initialize_weights(hidden_layer2_size, num_labels)

    X = theano.shared(initialize_weights(m-1, input_layer_size))
    y = theano.shared(np.mod(np.arange(0, m), num_labels))

    nn_params = np.hstack(
        [theta1.reshape(-1), theta2.reshape(-1), theta3.reshape(-1)]
    )

    nn = nnet(
        input_layer_size, [hidden_layer1_size, hidden_layer2_size],
        num_labels, lamda=lmbd
    )
    J, grad = nn._cost_and_gradient(X, y)

    J = theano.function([], J)
    grad = theano.function([], grad)

    def f(theta):
        nn.theta = theta
        return J()

    def g(theta):
        nn.theta = theta
        return grad()

    gradient = g(nn_params)
    numgrad = compute_numerical_gradient(f, nn_params)

    print(np.vstack([gradient, numgrad]).T)
    print(linalg.norm(gradient-numgrad) / linalg.norm(gradient+numgrad))


if __name__ == '__main__':
    check_NNGradients(10)
