# -*- coding: utf-8 -*-
import numpy as np


def sgd(blocks, f, x0, fprime, lamda=0, eta0=1e-5,
        maxiter=100, earlystop=None, patience=None,
        improvement_threshold=1e-4, callback=None):
    """Minimize a function using a simple stochastic gradient descent.

    Parameters
    ------
    blocks: int
        The number of training data blocks.

    f: callable, f(x, i)
        Objective function to be minimized. x is a 1d array of the variables
        that are to be changed in search for a minimun. i is block index
        determining which block to be used for computering the value of f.

    x0: 1d array
        Initial value of the optimal value x.

    fprime: callable, fprime(x, i)
        A function that compute the gradient of f w.r.t x. The return value
        should have the same shape of x.

    lamda: float
        Regularzation parameter.

    eta0: float
        Inital learning rate.

    maxiter: int
        Maximum number of iterations to perform.

    earlystop: callable, earlystop(x)
        Optional function that compute the validation cost over validation set.
        This will be used to prevent overfitting training set.

    patience: int
        Earlystop parameter. Look at this many blocks regardless.

    improvement_threshold: float
        Earlystop parameter. Increase patience when the decreaseing of
        validation cost is bigger than this value.

    callback: callable, callback(x, epoch)
        Optional function that will be called after each iteration.
    """

    def train_one(indx, eta):
        nonlocal theta
        theta = theta - eta * fprime(theta, indx)

    assert eta0 > 0
    t = 0
    theta = x0
    imin = 0
    imax = blocks - 1

    if earlystop:
        # early-stopping parameters
        patience = patience or max(100, blocks*20)
        patience_increase = 2  # wait this much longer when a new best is found
        # go through this many
        # blocks before checking the cost on the validation set
        validation_frequency = min(blocks, patience / 2)
        best_validation_loss = np.inf

    epoch = 0
    done = False
    while (epoch <= maxiter) and (not done):
        epoch += 1
        for i in range(imin, imax+1):
            eta = eta0 / (1 + lamda * eta0 * t)
            train_one(i, eta)
            t = t + 1
            if earlystop:
                iters = (epoch - 1) * blocks + i
                if (iters + 1) % validation_frequency == 0:
                    this_validation_loss = earlystop(theta)

                    if this_validation_loss < best_validation_loss:
                        # improve patience if loss improvement is good enough
                        loss = best_validation_loss - this_validation_loss
                        if loss > improvement_threshold:
                            patience = max(patience, iters * patience_increase)
                        best_validation_loss = this_validation_loss

        if patience < iters:
            done = True
        if callback:
            callback(theta, epoch)

    return theta
