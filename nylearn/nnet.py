# -*- coding: utf-8 -*-
import numpy as np
import theano
import theano.tensor as T

from nylearn.base import Layer
from nylearn.logistic import LogisticRegression
from nylearn.utils import shared

sigmoid = T.nnet.sigmoid


class nnet:
    """

    Parameters:
    ------
    inpute_layer: int
        The number of input layer units.

    hidden_layers: list
        A list of numbers of units in each hidden layers.

    output_layer: int
        The number of output layer units.

    lamda: float
        Parameter used for regularization. Set 0 to disable regularize.
    """
    def __init__(self, input_layer_size, hidden_layers_size,
                 output_layer_size, lamda=0):
        assert len(hidden_layers_size) > 0

        self.lamda = shared(lamda, name='lamda')

        self.layers_size = [input_layer_size, output_layer_size]
        self.layers_size[1:-1] = hidden_layers_size
        self.hidden_layers = [
            Layer(self.layers_size[i-1], self.layers_size[i], self.lamda)
            for i in range(1, len(self.layers_size)-1)
        ]
        self.output_layer = LogisticRegression(
            self.layers_size[-2], self.layers_size[-1], self.lamda.get_value()
        )
        self.layers = self.hidden_layers + [self.output_layer]

    @property
    def theta(self):
        return np.hstack([l.theta for l in self.layers])

    @theta.setter
    def theta(self, val):
        start = 0
        for i, l in enumerate(self.layers):
            end = start + (self.layers_size[i]+1) * self.layers_size[i+1]
            l.theta = val[start:end]
            start = end

    def _feedforward(self, X):
        o = Layer.add_bias(X)
        for hidden in self.hidden_layers:
            hidden.input = o
            hidden.value = hidden.input.dot(hidden._theta)
            hidden.output = Layer.add_bias(T.nnet.sigmoid(hidden.value))
            o = hidden.output
        self.output_layer.input = o

    def _backpropagation(self, y):
        m = y.shape[0]

        py = self.output_layer._p_given_X(self.hidden_layers[-1].output)
        y = T.eq(
            T.shape_padleft(T.arange(self.layers_size[-1])),
            T.shape_padright(y))
        dy = py - y  # output layer de/dy = py - y
        delta = dy  # de/dz = de/dy because y = z
        self.output_layer.grad = self.output_layer.input.T.dot(delta) / m
        _theta = self.output_layer._theta[1:]

        for hidden in self.hidden_layers:
            dy = delta.dot(_theta.T)  # de/dy = w * de/dz(delta)
            o = hidden.output[:, 1:]
            delta = o * (1 - o) * dy  # de/dz = g' * de/dy
            hidden.grad = hidden.input.T.dot(delta) / m
            _theta = hidden._theta[1:]

    def _cost_and_gradient(self, X, y):
        y = y.flatten()  # make sure that y is a vector
        m = y.shape[0]

        self._feedforward(X)
        self._backpropagation(y)

        reg_cost_list = []
        reg_grad_list = []
        for l in self.layers:
            c, g = l._l2_regularization(m)
            reg_cost_list.append(c)
            reg_grad_list.append((l.grad + g).flatten())
        J = self.output_layer._cost(self.output_layer.input, y) + \
            T.sum(reg_cost_list)
        grad = T.concatenate(reg_grad_list)

        return J, grad

    def predict(self, dataset):
        """Return predicted class labels as a numpy vecter

        Parameters
        ------
        dataset: dataset
        """
        y = theano.function([], self._predict_y(dataset.X))
        return y()

    def errors(self, dataset):
        """Return a float representing the error rate

        Parameters
        ------
        dataset: dataset
        """
        e = theano.function([], self._errors(dataset.X, dataset.y))
        return e()

    def _predict_y(self, X):
        o = X
        for hidden in self.hidden_layers:
            o = T.nnet.sigmoid(Layer.add_bias(o).dot(hidden._theta))

        return self.output_layer._predict_y(o)

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