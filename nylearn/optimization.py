# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import norm


def mbgd(blocks, f, x0, fprime, lamda=0, eta0=1e-5,
         maxiter=100, earlystop=None, patience=None,
         improvement_threshold=0.995, momentum=None, callback=None):
    """Minimize a function using a simple minibatch gradient descent.

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

    eta: float
        Learning rate.

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
        nonlocal inc

        inc = current_momentum * inc - eta * fprime(theta, indx)
        theta = theta + inc

    assert eta0 > 0
    theta = x0
    imin = 0
    imax = blocks - 1

    # early-stopping parameters
    patience = patience or max(100, blocks*20)
    if earlystop:
        patience_increase = 2  # wait this much longer when a new best is found
        # go through this many
        # blocks before checking the cost on the validation set
        validation_frequency = min(blocks, patience / 2)
        best_validation_loss = np.inf
        best_theta = None

    epoch = 0
    done = False
    current_momentum = 0
    inc = 0
    eta = eta0
    while (epoch <= maxiter) and (not done):
        if momentum:
            current_momentum = momentum(epoch)
        epoch += 1
        for i in range(imin, imax+1):
            iters = (epoch - 1) * blocks + i
            train_one(i, eta)

            if earlystop:
                if (iters + 1) % validation_frequency == 0:
                    this_validation_loss = earlystop(theta)

                    if this_validation_loss < best_validation_loss:
                        # improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss*improvement_threshold:
                            patience = max(patience, iters * patience_increase)
                            best_validation_loss = this_validation_loss
                        best_theta = np.copy(theta)

        if earlystop and patience < iters:
            done = True
        if callback:
            callback(theta, epoch)

    return best_theta, theta


# def S_ALAP(grad, last_grad, eta, u, mu=0.9, ro=0.01):
#     update_u = mu * u + (1 - mu) * grad**2
#     update_eta = eta * T.maximum(0.5, 1 + ro * (grad * last_grad) / update_u)

#     update = theano.function([], updates=[(u, update_u), (eta, update_eta)])
#     update()
