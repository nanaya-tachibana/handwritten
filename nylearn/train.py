# -*- coding: utf-8 -*-
import numpy as np
import theano
import theano.tensor as T


class gradient_descent_cg(object):

    def __init__(self, model, dataset):


    def train(self, model, dataset):
        data_X = dataset['X']
        data_y = dataset['y']
        batch_size = dataset['batch_size']
        batch_offset = theano.shared(0)
        batch_number = np.ceil(data_y.shape[0] / batch_size)

        dX = theano.shared(np.asarray(data_X, dtype=theano.config.floatX),
                           borrow=True)
        shared_y = theano.shared(np.asarray(data_y,
                                            dtype=theano.config.floatX),
                                 borrow=True)

        dy = T.cast(shared_y, 'int32')
        X = T.matrix('X')
        y = T.ivector('y')
        J, grad = model._cost_function(X, y)
        offset = T.lscalar()
        batch_cost = theano.function(
            [offset], J, givens={
                X: dX[offset*batch_size:(offset+1)*batch_size],
                y: dy[offset*batch_size:(offset+1)*batch_size]
            }, name='batch_cost')
        batch_grad = theano.function(
            [offset], grad, givens={
                X: dX[offset*batch_size:(offset+1)*batch_size],
                y: dy[offset*batch_size:(offset+1)*batch_size]
            }, name='batch_grad')

        def f(theta):
            model.theta = theta
            return batch_cost(batch_offset.get_value())

        def g(theta):
            model.theta = theta
            return batch_grad(batch_offset.get_value())

        def callback(theta):
            of = batch_offset.get_value()
            batch_offset.set_value((of + 1) % batch_number)

        from scipy.optimize import fmin_cg

        model.theta = fmin_cg(f=f, x0=model.theta, fprime=g,
                              callback=callback, maxiter=100)
