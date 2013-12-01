# -*- coding: utf-8 -*-
from functools import wraps
import numpy as np
from .base import Base
from .utils import sigmoid
from .optimize import mini

class LogisticRegression(Base):
    """Logistic regression classifier

    Uses one-vs.-all(OvA) for multiclass case.

    Parameters
    ------
    labels: list
    A list of labels used in training data.

    lmbd: float
    Use for regularization. Set 0 to disable regularize.
    """

    def __init__(self, labels, lmbd=0):
        super(LogisticRegression, self).__init__(lmbd=lmbd)
        self.labels = labels

    def _read_cache(self, theta):
        """Hack for func J and grad in _cost_function sharing data"""
        value = self.cache
        if value == None or value[0] != theta:
            _theta = self.ignore_bias(theta)
            value = (theta, _theta)
            self.cache = value
        return value[1]

    def _cost_function(self, X, y):
        m, X = self.add_bias(X)

        def J(theta):
            """Compute the penalize of using @theta for predicting"""
            _theta = self._read_cache(theta)
            return  (
                - y.dot(np.log(sigmoid(X.dot(theta))))
                - (1-y).dot(np.log(1-sigmoid(X.dot(theta))))
            ) / m + self.lmbd * _theta.dot(theta) / (2*m)

        def grad(theta):
            """Compute the gradient of the cost w.r.t. to the @theta"""
            _theta = self._read_cache(theta)
            return X.T.dot(sigmoid(X.dot(theta)) - y) / m + self.lmbd * _theta / m

        return J, grad, theta

    def train(self, X, y, maxiter=50, theta=None, method='BFGS'):
        """Use data @X @y to train the classifier

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
            guess = np.zeros(X.shape[1] + 1)
        assert len(guess) == X.shape[1] + 1

        if len(self.labels) <= 2:
            J, grad = self._cost_function(X, y)
            self.param = mini(J, grad, maxiter, guess, method=method)
        else:
            t = []
            for label in self.labels:
                J, grad = self._cost_function(X, np.array(y == label, dtype=np.float64))
                t.append(mini(J, grad, maxiter, guess, method=method))
            self.param = np.array(t).T

    def predict(self, X):
        """Return predicted classes.

        Parameters
        ------
        X: ndarray
        samples
        """
        m, X = self.add_bias(X)
        if len(self.labels) <= 2:
            return X.dot(self.param) > 0.5
        return X.dot(self.param).argmax(axis=1)
