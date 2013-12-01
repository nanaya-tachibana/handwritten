#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy import linalg
from nyml.neural_network import NeuralNetworkClassifier

def initialize_weights(fout, fin):
    return np.sin(np.arange(1, fout * (fin+1)+1)).reshape((fout, fin+1)) / 10

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
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    theta1 = initialize_weights(hidden_layer_size, input_layer_size)
    theta2 = initialize_weights(num_labels, hidden_layer_size)

    X  = initialize_weights(m, input_layer_size - 1)
    y  = np.mod(np.arange(0, m), num_labels)

    nn_params = np.hstack([theta1.reshape(-1), theta2.reshape(-1)])

    nn = NeuralNetworkClassifier(
        list(range(num_labels)), input_layer_size, [hidden_layer_size], lmbd=lmbd
    )
    J, grad = nn._cost_function(X, y)

    g = grad(nn_params)
    numgrad = compute_numerical_gradient(J, nn_params)

    print(np.vstack([g, numgrad]).T)
    print(linalg.norm(g-numgrad) / linalg.norm(g+numgrad))


if __name__ == '__main__':
    check_NNGradients(3)
