# -*- coding: utf-8 -*-
import numpy as np
from utils import sigmoid

class Base():
    param = None
    def __init__(self, lmbd=0):
        """

        Parameters
        ------
        lmbd: float
        use for regularization
        """

        self.lmbd = lmbd

    def preprocess(self, X, theta):
        m, X = self.add_bias(X)
        _theta = theta.copy()
        _theta[0] = 0
        return m, X, _theta

    def add_bias(self, X):
        m = X.shape[0]
        X = np.hstack((np.ones((m,1), dtype=X.dtype), X))
        return m, X


class LogisticRegression(Base):
    """Logistic regression classifier

    Uses one-vs.-all(OvA) for multiclass case.
    """

    def __init__(self, labels, lmbd=0):
        super(LogisticRegression, self).__init__(lmbd=lmbd)
        self.labels = labels

    def _cost(self, X, y, theta):
        """Compute the penalize of using @theta for regression"""
        m, X, _theta = self.preprocess(X, theta)
        return  (
            - y.dot(np.log(sigmoid(X.dot(theta))))
            - (1-y).dot(np.log(1-sigmoid(X.dot(theta))))
        ) / m + self.lmbd * _theta.dot(theta) / (2*m)

    def _grad(self, X, y, theta):
        """Computer the gradient of the cost w.r.t. to the @theta"""
        m, X, _theta = self.preprocess(X, theta)
        return X.T.dot(sigmoid(X.dot(theta)) - y) / m + self.lmbd * _theta / m

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
