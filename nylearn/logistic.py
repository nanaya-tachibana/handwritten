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

    def _cost(self, X, y):
        return -T.mean(T.log(self._p_given_X(X))[T.arange(y.shape[0]), y])

    def _cost_and_gradient(self, X, y):
        """Compute penalize and gradient for current @theta"""
        X, y = self.preprocess(X, y)
        m = y.shape[0]

        reg = self._l2_regularization(m)
        J = self._cost(X, y) + reg
        grad = self._gradient(J)

        return J, grad

    def preprocess(self, X, y=None):
        if y is not None:
            y = y.flatten()  # make sure that y is a vector
        X = self.add_bias(X)
        return X, y

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

    def _p_given_X(self, X):
        """Compute `p(y = i | x)` corresponding to the output."""
        return T.nnet.softmax(X.dot(self._theta))

    def _predict_y(self, X):
        """Predict y given x by choosing `argmax_i P(Y=i|X, theta)`.

        Parameters
        ------
        X: tensor like
            feature matrix
        """
        return T.argmax(self._p_given_X(X), axis=1)

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
