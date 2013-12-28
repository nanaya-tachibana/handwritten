# -*- coding: utf-8 -*-
import theano.tensor as T

from nylearn.utils import shared, nn_random_paramters


class Layer:

    theta = property(
        lambda self: self._theta.get_value(borrow=True).reshape(-1),
        lambda self, val: self._theta.set_value(
            val.reshape(self.n_in+1, self.n_out), borrow=True
        )
    )

    def __init__(self, n_in, n_out, lamda):
        """
        n_in: int
            Number of input units, the dimension of the space
            in which the datapoints lie.

        n_out: int
            Number of output units, the dimension of the space
            in which the labels lie.

        lamda: theano like
            Parameter used for regularization. Set 0 to disable regularize.
        """
        self.n_in = n_in
        self.n_out = n_out
        self.lamda = lamda

        self._theta = shared(self._initialize_theta())
        self.w = self._theta[1:, :]
        self.b = self._theta[0, :]

    def _initialize_theta(self):
        """Override this function to use other initial value"""
        return nn_random_paramters(self.n_in+1, self.n_out)

    def _gradient(self, cost):
        return T.grad(cost, self._theta).flatten()

    def _l2_regularization(self, m):
        w = self.w
        lamda = self.lamda

        return lamda/(2*m) * T.sum(w**2)
        #reg_grad = T.concatenate([T.zeros((1, w.shape[1])), lamda * w / m])

    @classmethod
    def add_bias(cls, X):
        return T.concatenate([T.ones((X.shape[0], 1)), X], axis=1)
