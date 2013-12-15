# -*- coding: utf-8 -*-
import theano
import theano.tensor as T

from nylearn.base import Layer


class LogisticRegression(Layer):
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
        super(LogisticRegression, self).__init__(n_in, n_out, lamda=lamda)

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

    def _cost_function(self, X, y):
        """Compute penalize and gradient for current @theta"""

        m = X.shape[0]
        X = self.add_bias(X)

        J = -T.mean(T.log(self._p_given_X(X))[T.arange(m), y])
        grad = T.grad(J, self._theta)
        rj, rg = self._l2_regularization(m)

        return J+rj, (grad+rg).flatten()

    def _predict_y(self, X):
        """Predict y given x by choosing `argmax_i P(Y=i|X, theta)`.

        Parameters
        ------
        X: tensor like
            feature matrix
        """
        return T.argmax(self._p_given_X(self.add_bias(X)), axis=1)

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
