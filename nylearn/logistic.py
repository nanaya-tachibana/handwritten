# -*- coding: utf-8 -*-
import numpy as np
import theano
import theano.tensor as T

from nylearn.base import Base


class LogisticRegression(Base):
    """Multi-class logistic regression classifier

    Use one-vs.-all(OvA) for multiclass case.

    Parameters
    ------
    n_in: int
        Number of input units, the dimension of the space
        in which the datapoints lie.

    n_out: int
        Number of output units, the dimension of the space
        in which the labels lie.

    lamda: float
        Parameter used for regularization. Set 0 to disable regularize.
    """

    def __init__(self, n_in, n_out, lamda=0):
        super(LogisticRegression, self).__init__(lamda=lamda)
        self.n_in = n_in + 1
        self.n_out = n_out
        self._theta = theano.shared(
            value=np.zeros((self.n_in, self.n_out), dtype=theano.config.floatX),
            name='theta'
        )

    @property
    def theta(self):
        return self._theta.get_value().reshape(-1)

    @theta.setter
    def theta(self, value):
        self._theta.set_value(value.reshape((self.n_in, self.n_out)))

    def _p_given_X(self, X):
        """Compute `p(y = i | x)` corresponding to the output."""
        return T.nnet.softmax(X.dot(self._theta))

    def _cost_function(self, X, y):
        """Compute penalize and gradient for current @theta"""

        theta = self._theta
        _theta = self.ignore_bias(theta)
        m = y.shape[0]
        n = self.n_out
        X = self.add_bias(X)

        J = -T.mean(T.log(self._p_given_X(X))[T.arange(m), y])
        grad = T.grad(J, theta)
        rj, rg = self._l2_reg(_theta, m, n)

        return J+rj, (grad+rg).flatten(1)

    def predict(self, X):
        """Return predicted classes.

        Predicts y given x by choosing `argmax_i P(Y=i|x,W,b)`.

        Parameters
        ------
        X: ndarray
        samples
        """
        return T.argmax(self._p_given_X(self.add_bias(X)), axis=1)

    # def train(self, X, y, maxiter=50, theta=None, method='BFGS'):
    #     """Use data @X @y to train the classifier

    #     Parameters
    #     ------
    #     X: ndarray
    #     Samples

    #     y: ndarray
    #     Sample labels

    #     maxiter: integer
    #     Maximum number of iterations to perform

    #     theta: ndarray
    #     Initial estimate

    #     method: string
    #     Training method
    #     """
    #     guess = theta
    #     if not guess:
    #         guess = np.zeros(X.shape[1] + 1)
    #     assert len(guess) == X.shape[1] + 1

    #     if len(self.labels) <= 2:
    #         J, grad = self._cost_function(X, y)
    #         self.param = mini(J, grad, maxiter, guess, method=method)
    #     else:
    #         t = []
    #         for label in self.labels:
    #             J, grad = self._cost_function(X, np.array(y == label, dtype=np.float64))
    #             t.append(mini(J, grad, maxiter, guess, method=method))
    #         self.param = np.array(t).T
