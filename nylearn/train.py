# -*- coding: utf-8 -*-
import theano


class gradient_descent_cg(object):

    def __init__(self, model):
        self.model = model

    def train(self, dataset, maxiter):

        J, grad = self.model._cost_function(dataset.X, dataset.y)
        cost_func = theano.function([], J, name='batch_cost')
        grad_func = theano.function([], grad, name='batch_grad')

        def f(theta):
            self.model.theta = theta
            return cost_func()

        def g(theta):
            self.model.theta = theta
            return grad_func()

        def callback(theta):
            print(cost_func())

        from scipy.optimize import fmin_cg

        self.model.theta = fmin_cg(f=f, x0=self.model.theta, fprime=g,
                                   callback=callback, maxiter=maxiter)
