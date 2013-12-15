# -*- coding: utf-8 -*-
import numpy as np
import theano.tensor as T

from nylearn.utils import shared


class Layer(object):

    theta = property(
        lambda self: self._theta.get_value().reshape(-1),
        lambda self, val: self._theta.set_value(
            val.reshape(self.n_in+1, self.n_out)
        )
    )

    def __init__(self, n_in, n_out, lamda=0):
        """
        """
        self.lamda = shared(lamda, name='lamda')
        self.n_in = n_in
        self.n_out = n_out

        self._theta = shared(self._initialized_theta())
        self.w = self._theta[1:, :]
        self.b = self._theta[0, :]

    def _initialized_theta(self):
        """Overwrriten this function to use other initial value"""
        return np.zeros((self.n_in+1, self.n_out))

    def _l2_regularization(self, m):
        w = self.w
        lamda = self.lamda

        reg_cost = lamda * (w**2).mean() / 2
        reg_grad = T.concatenate([T.zeros((1, w.shape[1])), lamda * w / m])

        return reg_cost, reg_grad

    def add_bias(self, X):
        return T.concatenate([T.ones((X.shape[0], 1)), X], axis=1)
