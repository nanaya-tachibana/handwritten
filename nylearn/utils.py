# -*- coding: utf-8 -*-
import numpy as np
from sympy import *
from sympy.printing.theanocode import theano_function as tfunction


def random_parameters(shape, epsilon):
    """Return a matrix with values in [-@epsilon,@epsilon)"""
    return np.random.random(shape)*2*epsilon - epsilon


def _sigmoid():
    x = symbols('x')
    return 1 / (1 + exp(-x))

def sigmoid(z):
    """Return the sigmoid of @z

    Parameters
    ------
    z: scalar or ndarray
    return value type depended on z
    """
    dim = len(z.shape) if isinstance(z, ndarray) else 0

    f = tfunction([x], [_sigmoid()], dims={x: dim}, dtypes={x: 'float64'})

    return f(z)


def sigmoid_gradient(z):
    """Return gradient of the sigmoid function given @z

    Parameters
    ------
    z: scalar or ndarray
    return value type depended on z
    """
    x = sigmoid(z)
    return x * (1 - x)
