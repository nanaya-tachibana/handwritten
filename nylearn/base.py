# -*- coding: utf-8 -*-
import theano
import theano.tensor as T


class Base():

    def __init__(self, lamda=0):
        """
        """
        self.lamda = theano.shared(lamda, name='lamda')

    def _l2_reg(self, _theta, m, n):
        reg_cost = self.lamda * (_theta**2).mean() / 2
        reg_grad = self.lamda * (
            T.concatenate([T.zeros((1, n)), _theta], axis=0) / m
        )
        return reg_cost, reg_grad

    def ignore_bias(self, theta):
        return theta[1:, :]

    def add_bias(self, X):
        return T.concatenate([T.ones((X.shape[0], 1)), X], axis=1)
