# -*- coding: utf-8 -*
import operator
from functools import reduce
import numpy as np
from scipy import linalg

from .base import Base
from .utils import sigmoid, sigmoid_gradient, random_parameters
from .optimize import mini

class NeuralNetwork(Base):
    """Neural Network Model

    """

    def __init__(self, input_layer_size, output_layer_size, hidden_layer_size=None, lmbd=0):
        super(NeuralNetwork, self).__init__(lmbd=lmbd)
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size

    def _read_cache(self, theta, X, y):
        """Hack for func J and grad in _cost_function sharing data"""
        value = self.cache
        if value == None or value[0] != theta:
            theta = self.reshape_theta(theta, self.layer_size)
            _theta = [self.ignore_bias(t) for t in theta]
            a, z = self.feedforward(theta, X)
            d = self.backpropagation(theta, a, z, y)
            value = (theta, _theta, a, d)
            self.cache = value
        return value

    def _cost_function(self, X, y):
        m, X, y = self.preprocess(X, y)

        def J(theta):
            """Compute the penalize of using @theta for predicting"""
            theta, _theta, a, d = self._read_cache(theta, X, y)
            return (
                self.output_layer_cost(m, a[-1], y) + self.lmbd * reduce(
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
        Sample values

        maxiter: integer
        Maximum number of iterations to perform

        theta: ndarray
        Initial estimate

        method: string
        Training method
        """
        assert X.shape[1] == self.input_layer_size
        guess = theta
        if not guess:
            guess = self.initialize_theta(self.layer_size)

        J, grad = self._cost_function(X, y)
        self.param = mini(J, grad, maxiter, guess, method=method)
        self.param = self.reshape_theta(self.param, self.layer_size)

    def preprocess(self, X, y):
        """Overwritten this function to perform addition preprocess"""
        m, X = self.add_bias(X)
        return m, X, y

    def output_layer_hypotheses(self, a, theta):
        """OutputLayer hypotheses function. Overwritten this function!"""
        return a.dot(theta.T)

    def output_layer_cost(self, m, a, y):
        """Compute outputlayer cost. Overwritten this function!"""
        return np.zeros(len(y))

    def initialize_theta(self, layer_size):
        """Return all thetas with values random initalized as a list

        Paramters
        ------
        layer_size: list
        units number in each layer
        """
        return np.hstack(
            random_parameters(
                (layer_size[i+1], layer_size[i]+1),
                np.sqrt(6) / np.sqrt(layer_size[i]+layer_size[i+1])
            ).reshape(-1) for i in range(len(layer_size)-1)
        )

    def reshape_theta(self, theta, layer_size):
        """According to @layer_size reshape vector @theta to matrix"""
        offset = 0
        theta_list = []

        for i in range(len(layer_size)-1):
            end = offset + (layer_size[i]+1) * layer_size[i+1]
            theta_list.append(
                theta[offset:end].reshape((layer_size[i+1], layer_size[i]+1))
            )
            offset = end

        return theta_list

    def feedforward(self, theta, X):
        a = [X] # input layer. a_0 = X + bias
        z = [0]

        for i in range(1, len(self.hidden_layer_size)+1):
            z.append(a[i-1].dot(theta[i-1].T))  # z_i = a_(i-1) * theta_(i-1)
            a.append(self.add_bias(sigmoid(z[i]))[1])  # a_i = sigmoid(z_i) + bias
        a.append(self.output_layer_hypotheses(a[-1], theta[-1])) # output layer.

        return a, z

    def backpropagation(self, theta, a, z, y):
        d = [a[-1] - y]  # output layer. d_n = a_n - y

        for i in range(len(self.hidden_layer_size), 0, -1):
            d[0:0] = [d[-1].dot(theta[i][:, 1:]) * sigmoid_gradient(z[i])]
        d[0:0] = [0]

        return d

    @property
    def layer_size(self):
        """Return layers's units number as a list from input layer to output layer"""
        if not self.hidden_layer_size:
            self.hidden_layer_size = [self.input_layer_size]

        s = [self.input_layer_size]
        s.extend(self.hidden_layer_size)
        s.append(self.output_layer_size)

        return s


class NeuralNetworkClassifier(NeuralNetwork):
    """Neural Network Classifier

    Parameters
    ------
    labels: list
    A list of labels used in training data.

    input_layer_size: integer
    Input layer units number.

    hidden_layer_size: list
    A list of hidden layer size.
    If None, use one hidden layer with default setting.

    lmbd: float(>=0)
    Use for regularization. Set 0 to disable regularization.
    """
    def __init__(self, labels, input_layer_size, hidden_layer_size=None, lmbd=0):
        super(NeuralNetworkClassifier, self).__init__(
            input_layer_size, len(labels), hidden_layer_size, lmbd
        )
        self.labels = labels

    def preprocess(self, X, y):
        m, X, y = super(NeuralNetworkClassifier, self).preprocess(X, y)

        y = np.array(
            np.tile(np.array(self.labels), (m,1)) == y[:, np.newaxis],
            dtype=np.float64
        )

        return m, X, y

    def output_layer_hypotheses(self, a, theta):
        return sigmoid(a.dot(theta.T))

    def output_layer_cost(self, m, a, y):
        return (
            - y * np.log(a)
            - (1-y) * np.log(1-a)
        ).sum() / m

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

        return a.argmax(axis=1)

class NeuralNetworkRegression(NeuralNetwork):
    """Neural Network Regression

    Parameters
    ------
    input_layer_size: integer
    Input layer units number.

    hidden_layer_size: list
    A list of hidden layer size.
    If None, use one hidden layer with default setting.

    lmbd: float(>=0)
    Use for regularization. Set 0 to disable regularization.
    """
    def __init__(self, input_layer_size, hidden_layer_size=None, lmbd=0):
        super(NeuralNetworkRegression, self).__init__(
            input_layer_size, 1, hidden_layer_size, lmbd
        )

    def output_layer_hypotheses(self, a, theta):
        return a.dot(theta.T)

    def output_layer_cost(self, m, a, y):
        return linalg.norm(a - y) / (2*m)

    def predict(self, X):
        """Return predicted values.

        Parameters
        ------
        X: ndarray
        samples
        """
        a = X

        for theta in self.param[:-1]:
            a = sigmoid(self.add_bias(a)[1].dot(theta.T))

        return self.add_bias(a)[1].dot(self.param[-1].T)
