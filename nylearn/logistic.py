# -*- coding: utf-8 -*-
import numpy as np
import theano
import theano.tensor as T

from nylearn.base import Layer
from nylearn.utils import shared


class LogisticRegression(Layer):
    """Multi-class logistic regression classifier

    Use one-vs.-all(OvA) for multiclass case.

    Parameters
    ------
    features: int
        Number of input sample's features.

    labels: int
        Number of output classes.

    lamda: float
        Parameter used for regularization. Set 0 to disable regularize.
    """

    def __init__(self, features, labels, lamda=0):
        lamda = shared(lamda, name='lamda')
        super(LogisticRegression, self).__init__(features, labels, lamda)

    def _initialize_theta(self):
        """Override this function to use other initial value"""
        return np.zeros((self.n_in+1, self.n_out))

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

    def _p_given_X(self, X):
        """Compute `p(y = i | x)` corresponding to the output."""
        return T.nnet.softmax(X.dot(self._theta))

    def _cost(self, X, y):
        return -T.mean(T.log(self._p_given_X(X))[T.arange(y.shape[0]), y])

    def _cost_and_gradient(self, X, y):
        """Compute penalize and gradient for current @theta"""

        y = y.flatten()  # make sure that y is a vector
        m = y.shape[0]
        X = Layer.add_bias(X)

        rj = self._l2_regularization(m)
        J = self._cost(X, y) + rj
        grad = T.grad(J, self._theta)

        return J, grad.flatten()

    def _predict_y(self, X):
        """Predict y given x by choosing `argmax_i P(Y=i|X, theta)`.

        Parameters
        ------
        X: tensor like
            feature matrix
        """
        return T.argmax(self._p_given_X(Layer.add_bias(X)), axis=1)

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
