# -*- coding: utf-8 -*-
import numpy as np
import theano
import theano.tensor as T

from nylearn.dataset import Dataset


class MinibatchGradientDescent:
    """A class for minimizing function via stochastic gradient descent.

    Minibatch gradient split training set to serveral batches and compute
    each iteration use one batch to compute gradient, so it can train more
    efficient on large dataset.
    """
    def __init__(self, model):
        """
        Parameters
        ------
        model: object
            A Python object representing the model to train
        """
        self.model = model

    def train(self, dataset, maxiter=200, batch_size=500, eta=1e-5,
              validation_set=None, patience=None,
              improvement_threshold=0.995, momentum=None):
        """Train the given @model with @dataset.

        Parameters
        ------
        dataset: object
            A Dataset object containing training set.

        maxiter: int
            Maximum number of iterations to perform.

        batch_size: int
            The size of batches that the training data will be splited into.

        eta: float
            Inital learning rate. Changing this value can have a great impact
            on training precess. Too large value will make the reslut non-
            converge. Too samll value can make training time-consuming.

        validation_set: object, Optional
            A Dataset object containing validation set used for early-stopping.

        patience: int
            Earlystop parameter. Look at this many blocks regardless.

        improvement_threshold: float
            Earlystop parameter. Increase patience when the decreaseing of
            validation cost is bigger than this value.
        """

        def build_func(start, end, dataset, args=()):
            """Build functions that compute cost and gradient."""
            x = dataset.X[start:end]
            y = dataset.y[start:end]

            J, grad = self.model._cost_and_gradient(x, y)
            givens = {x: dataset.X[start:end], y: dataset.y[start:end]}
            f = theano.function(list(args), J, givens=givens)
            g = theano.function(list(args), grad, givens=givens)
            return f, g

        m = dataset.size
        assert m >= 1
        lamda = self.model.lamda.get_value()
        num_train_batches = int(np.ceil(m / batch_size))
        indx = T.lscalar()
        cost_func, grad_func = build_func(indx * batch_size,
                                          (indx + 1) * batch_size,
                                          dataset, (indx,))

        validation_cost = None
        earlystop = None
        if validation_set:
            assert isinstance(validation_set, Dataset)
            assert validation_set.size
            v_cost, _ = self.model._cost_and_gradient(
                validation_set.X, validation_set.y)
            validation_cost = theano.function([], v_cost)

            def es(theta):
                self.model.theta = theta
                return validation_cost()
            earlystop = es

        def f(theta, indx):
            self.model.theta = theta
            return cost_func(indx)

        def g(theta, indx):
            self.model.theta = theta
            return grad_func(indx)

        def callback(theta, epoch):
            dataset.permute()
            if validation_cost is not None:
                print('epoch', epoch, 'validation cost', validation_cost())

        from nylearn.optimization import mbgd

        self.model.theta, best = mbgd(num_train_batches, f, self.model.theta,
                                      g, lamda=lamda, eta0=eta, maxiter=maxiter,
                                      earlystop=earlystop, patience=patience,
                                      improvement_threshold=improvement_threshold,
                                      momentum=momentum, callback=callback)
        return best


class ConjugateGradientDescent:
    """A class for minimizing function via conjucate gradient descent."""

    def __init__(self, model):
        """
        Parameters
        ------
        model: object
            A Python object representing the model to train
        """
        self.model = model

    def train(self, dataset, maxiter=50):
        """Train the given @model with @dataset.

        Parameters
        ------
        dataset: object
            A Dataset object containing training set.

        maxiter: int
            Maximum number of iterations to perform.
        """
        J, grad = self.model._cost_and_gradient(dataset.X, dataset.y)
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


def momentum(init=0.9, final=0.9, start=0, end=0):
    """
    Implements momentum as described in Section 9 of
    "A Practical Guide to Training Restricted Boltzmann Machines",
    Geoffrey Hinton.

    Parameters
    ------
    init: float
        The momentum coefficient to use at the beginning of learning.
    final: float
        The momentum coefficient to use at the beginning of learning.
    start: int
        The epoch on which to start growing the momentum cofficient.
    end: int
        The epoch on which the momentum should reach its final value.

    To use a fixed momentum, setting @start = @end and
    @init to the fixed value.
    """
    assert start <= end
    inv = end - start

    def get_current_momentum(epoch):
        if inv == 0:
            return init
        else:
            alpha = (epoch - start) / inv
            if alpha < 0:
                alpha = 0
            elif alpha > 1:
                alpha = 1
            return init * (1 - alpha) + final * alpha

    return get_current_momentum
