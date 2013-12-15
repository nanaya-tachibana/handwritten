# -*- coding: utf-8 -*-
import numpy as np
import theano
import theano.tensor as T

from nylearn.dataset import Dataset


class StochasticGradientDescent:
    """A class for minimizing function via stochastic gradient descent.

    Stochastic gradient split training set to serveral batches and compute
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

    def train(self, dataset, maxiter=200, batch_size=500, alpha=1e-5,
              validation_set=None, patience=None, improvement_threshold=1e-4):
        """Train the given @model with @dataset.

        Parameters
        ------
        dataset: object
            A Dataset object containing training set.

        maxiter: int
            Maximum number of iterations to perform.

        batch_size: int
            The size of batches that the training data will be splited into.

        alpha: float
            Inital learning rate. Changing this value can have a great impact
            on training precess. Too large value will make the reslut non-
            converge. Too samll value can make training time-consuming.

        validation_set: object, Optional
            A Dataset object containing validation set used for early-stopping.
            If not given, 20% of training set will be used as validation set.

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

            J, grad = self.model._cost_function(x, y)
            givens = {x: dataset.X[start:end], y: dataset.y[start:end]}
            f = theano.function(list(args), J, givens=givens)
            g = theano.function(list(args), grad, givens=givens)
            return f, g

        def create_validation_set(m):
            x = int(np.floor(.2 * m))
            if x == 0:
                x = 1
            return x

        m = dataset.y.shape[0].eval()
        assert m >= 2
        lamda = self.model.lamda.get_value()

        offset = 0
        if validation_set is None:
            validation_set_size = create_validation_set(m)
            validation_set = dataset
            offset = validation_set_size
        else:
            assert isinstance(validation_set, Dataset)
            validation_set_size = validation_set.X.shape[0].eval()
        num_train_batches = int(np.ceil((m-offset) / batch_size))

        indx = T.lscalar()
        start = offset + indx * batch_size
        end = offset + (indx + 1) * batch_size
        cost_func, grad_func = build_func(start, end, dataset, (indx,))

        validation_cost, _ = build_func(0, validation_set_size, validation_set)

        def f(theta, indx):
            self.model.theta = theta
            return cost_func(indx)

        def g(theta, indx):
            self.model.theta = theta
            return grad_func(indx)

        def callback(theta, epoch):
            print("epoch ", epoch, "cost:", validation_cost())

        def earlystop(theta):
            return validation_cost()

        from nylearn.optimization import sgd

        self.model.theta = sgd(num_train_batches, f, self.model.theta, g,
                               lamda=lamda, eta0=alpha, maxiter=maxiter,
                               earlystop=earlystop, patience=patience,
                               improvement_threshold=improvement_threshold,
                               callback=callback)


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
