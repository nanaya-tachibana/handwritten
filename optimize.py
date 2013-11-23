# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import minimize

def mini(f, gradf, maxiter, guess, method='CG'):
    """Minimize the function @f use method @method

    Return a vector of parameters which minimize f.

    Parameters
    ------
    """
    opts = {'maxiter': maxiter}
    return minimize(
        f, guess, jac=gradf, method=method, options=opts
    ).x
