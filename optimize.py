# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import minimize

def mini(cls, X, y, steps=50, guess=None, method='BFGS'):
    """Minimize the function @f use method @method

    Return a vector of parameters which minimize f.

    Parameters
    ------
    """

    def f(x, *args):
        X, y = args
        return cls._cost(X, y, x)

    def gradf(x, *args):
        X, y = args
        return cls._grad(X, y, x)

    if guess == None:
        guess = np.zeros(X.shape[1] + 1)
    assert len(guess) == X.shape[1] + 1

    opts = {'maxiter': steps,
            'gtol': 1e-8
    }
    labels = getattr(cls, 'labels', None)
    if not labels or len(labels) <= 2:
        cls.param = minimize(
            f, guess, jac=gradf, args=(X,y), method=method, options=opts
        ).x
    else:
        cls.param =  np.vstack(
            minimize(f, guess, jac=gradf, method=method, options=opts,
                     args=(X,np.array(y == label, dtype=np.uint8))).x
            for label in labels
        ).T
