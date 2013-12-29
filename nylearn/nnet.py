# -*- coding: utf-8 -*-
import numpy as np
import theano
import theano.tensor as T

from nylearn.base import Layer
from nylearn.logistic import LogisticRegression
from nylearn.utils import shared

sigmoid = T.nnet.sigmoid
tanh = T.tanh


class HiddenLayer(Layer):

    def __init__(self, n_in, n_out, lamda, activation=tanh):
        super(HiddenLayer, self).__init__(n_in, n_out, lamda)
        self.activation = activation
        if activation == sigmoid:
            self.theta = self.theta*4

    def output(self, input):
        lin = input.dot(self._theta)
        if self.activation == tanh:
            val = 1.7159*tanh(2/3 * lin)
        else:
            val = self.activation(lin)
        return val


class MLP(Layer):
    """
    A class representing fully connected multilayer perceptron.
    """
    def __init__(self, input_size, hidden_layers_size, output_size,
                 output_layer=LogisticRegression, lamda=0):
        """
        Parameters:
        ------
        inpute_size: int
            The number of input layer units.

        hidden_layers_size: list
            A list of numbers of units in each hidden layer.

        output_size: int
            The number of output layer units.

        output_layer: object

        lamda: float
            Parameter used for regularization. Set 0 to disable regularize.
        """
        assert isinstance(hidden_layers_size, list) and \
            len(hidden_layers_size) > 0

        self.lamda = shared(lamda, name='lamda')

        self.layers_size = [input_size, output_size]
        self.layers_size[1:-1] = hidden_layers_size
        self.hidden_layers = [
            HiddenLayer(self.layers_size[i-1], self.layers_size[i],
                        self.lamda, activation=tanh)
            for i in range(1, len(self.layers_size)-1)
        ]
        self.output_layer = output_layer(
            self.layers_size[-2], self.layers_size[-1], self.lamda.get_value())
        self.layers = self.hidden_layers + [self.output_layer]

    @property
    def theta(self):
        return np.hstack([l.theta for l in self.layers])

    @theta.setter
    def theta(self, val):
        start = 0
        for l in self.layers:
            size = np.prod(l.theta.shape)
            end = start + size
            l.theta = val[start:end]
            start = end

    def _cost(self, X, y):
        return self.output_layer._cost(X, y)

    def _gradient(self, cost):
        # Theano is awesome!!!
        return T.concatenate([l._gradient(cost) for l in self.layers])

    def _l2_regularization(self, m):
        return T.sum([l._l2_regularization(m) for l in self.layers])

    def _feedforward(self, X):
        input = X
        for hidden in self.hidden_layers:
            input = hidden.output(self.add_bias(input))
        return input

    # def _backpropagation(self, y):
    #     m = y.shape[0]

    #     py = self.output_layer._p_given_X(self.hidden_layers[-1].output)
    #     y = T.eq(
    #         T.shape_padleft(T.arange(self.layers_size[-1])),
    #         T.shape_padright(y))
    #     dy = py - y  # output layer de/dy = py - y
    #     delta = dy  # de/dz = de/dy because y = z
    #     self.output_layer.grad = self.output_layer.input.T.dot(delta) / m
    #     _theta = self.output_layer._theta[1:]

    #     for hidden in reversed(self.hidden_layers):
    #         dy = delta.dot(_theta.T)  # de/dy = w * de/dz(delta)
    #         o = hidden.output[:, 1:]
    #         delta = o * (1 - o) * dy  # de/dz = g' * de/dy
    #         hidden.grad = hidden.input.T.dot(delta) / m
    #         _theta = hidden._theta[1:]

    def _cost_and_gradient(self, X, y):
        X, y = self.preprocess(X, y)
        m = y.shape[0]

        reg = self._l2_regularization(m)
        J = self._cost(X, y) + reg
        grad = self._gradient(J)

        return J, grad

    def preprocess(self, X, y=None):
        if y is not None:
            y = y.flatten()  # make sure that y is a vector
        output_layer_input = self.add_bias(self._feedforward(X))
        return output_layer_input, y

    def output(self, input):
        return self._predict_y(input)

    def predict(self, dataset):
        """Return predicted class labels as a numpy vecter

        Parameters
        ------
        dataset: dataset
        """
        X, _ = self.preprocess(dataset.X)
        y = theano.function([], self._predict_y(X))
        return y()

    def errors(self, dataset):
        """Return a float representing the error rate

        Parameters
        ------
        dataset: dataset
        """
        X, y = self.preprocess(dataset.X, dataset.y)
        e = theano.function([], self._errors(X, y))
        return e()

    def _predict_y(self, X):
        return self.output_layer.output(X)

    def _errors(self, X, y):
        """Compute the rate of predict_y_i != y_i

        Parameters
        ------
        X: tensor like
            feature matrix

        y: tensor like
            class label
        """
        return T.mean(T.neq(self._predict_y(X), y))
