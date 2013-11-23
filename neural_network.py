# -*- coding: utf-8 -*
import operator
from functools import reduce
import numpy as np

from base import Base
from utils import sigmoid, sigmoid_gradient, random_parameters
from optimize import mini

class NeuralNetwork(Base):
    """Neural Network classifier

    Parameters
    ------
    labels: list
    A list of labels used in training data.

    lmbd: float
    Use for regularization. Set 0 to disable regularize.

    hiddens: list
    A list of hidden layer size.
    If None, use one hidden layer with default setting.
    """

    def __init__(self, labels, hiddens=None, lmbd=0):
        super(NeuralNetwork, self).__init__(lmbd=lmbd)
        self.labels = labels
        self.hiddens = hiddens

    def _read_cache(self, theta, X, y):
        """Hack for func J and grad in _cost_function sharing data"""
        value = self.cache
        if value == None or value[0] != theta:
            theta = self.reshape_theta(theta, self.layer_units(X[:, 1:]))
            _theta = [self.ignore_bias(t) for t in theta]
            a, z = self.feedforward(theta, X)
            d = self.backpropagation(theta, a, z, y)
            value = (theta, _theta, a, d)
            self.cache = value
        return value

    def _cost_function(self, X, y):
        m, X = self.add_bias(X)
        y = np.array(
            np.tile(np.array(self.labels), (m,1)) == y[:, np.newaxis],
            dtype=np.float64
        )

        def J(theta):
            """Compute the penalize of using @theta for predicting"""
            theta, _theta, a, d = self._read_cache(theta, X, y)
            return  ((
                - y * np.log(a[-1])
                - (1-y) * np.log(1-a[-1])
            ).sum() / m + self.lmbd * reduce(
                operator.add,
                ((_t * t).sum() for _t, t in zip(_theta, theta))
            ) / (2*m))

        def grad(theta):
            """Computer the gradient of the cost w.r.t. to the @theta"""
            theta, _theta, a, d = self._read_cache(theta, X, y)
            return np.hstack(
                (d[i+1].T.dot(a[i]) / m + self.lmbd * _theta / m).reshape(-1)
                for i, _theta in enumerate(_theta)
            ).reshape(-1)

        return J, grad

    def train(self, X, y, maxiter=100, theta=None, method='CG'):
        """Use data @X @y to train classifier

        Parameters
        ------
        X: ndarray
        Samples

        y: ndarray
        Sample labels

        maxiter: integer
        Maximum number of iterations to perform

        theta: ndarray
        Initial estimate

        method: string
        Training method
        """
        guess = theta
        if not guess:
            guess = self.initialize_theta(self.layer_units(X))

        J, grad = self._cost_function(X, y)
        self.param = mini(J, grad, maxiter, guess, method=method)
        self.param = self.reshape_theta(self.param, self.layer_units(X))

    def initialize_theta(self, layer_units):
        """Return all thetas with values random initalized as a list

        Paramters
        ------
        layer_units: list
        units number in each layer
        """
        return np.hstack(
            random_parameters(
                (layer_units[i+1], layer_units[i]+1),
                np.sqrt(6) / np.sqrt(layer_units[i]+layer_units[i+1])
            ).reshape(-1) for i in range(len(layer_units)-1)
        )

    def reshape_theta(self, theta, layer_units):
        offset = 0
        theta_list = []
        for i in range(len(layer_units)-1):
            end = offset + (layer_units[i]+1) * layer_units[i+1]
            theta_list.append(
                theta[offset:end].reshape((layer_units[i+1], layer_units[i]+1))
            )
            offset = end
        return theta_list

    def feedforward(self, theta, X):
        a = [X] # input layer. a_0 = X + bias
        z = [0]

        for i in range(1, len(self.hiddens)+1):
            z.append(a[i-1].dot(theta[i-1].T))  # z_i = a_(i-1) * theta_(i-1)
            a.append(self.add_bias(sigmoid(z[i]))[1])  # a_i = sigmoid(z_i) + bias
        z.append(a[-1].dot(theta[-1].T))  # output layer.
        a.append(sigmoid(z[-1]))     # a_n = sigmoid(z_n)

        return a, z

    def backpropagation(self, theta, a, z, y):
        d = [a[-1] - y]  # output layer. d_n = a_n - y

        for i in range(len(self.hiddens), 0, -1):
            d[0:0] = [d[-1].dot(theta[1][:, 1:]) * sigmoid_gradient(z[1])]
        d[0:0] = [0]
        return d

    def layer_units(self, X):
        """Return layers's units number as a list from input layer to output layer"""
        if not self.hiddens:
            self.hiddens = [X.shape[1]]
        s = [X.shape[1]]
        s.extend(self.hiddens)
        s.append(len(self.labels) if len(self.labels) > 2 else 1)
        return s

    def predict(self, X):
        """Return predicted classes.

        Parameters
        ------
        X: ndarray
        samples
        """
        a = X
        for theta in self.param:
            a = sigmoid(self.add_bias(a)[1].dot(theta.T))
        if len(self.labels) <= 2:
            return a > 0.5
        return a.argmax(axis=1)
