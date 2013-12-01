# -*- coding: utf-8 -*-
import numpy as np

def random_parameters(shape, epsilon):
    """Return a matrix with values in [-@epsilon,@epsilon)"""
    return np.random.random(shape)*2*epsilon - epsilon

def sigmoid(z):
    """Return the sigmoid of @z

    Parameters
    ------
    z: scalar or ndarray
    return value type depended on z
    """
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_gradient(z):
    """Return gradient of the sigmoid function given @z

    Parameters
    ------
    z: scalar or ndarray
    return value type depended on z
    """
    x = sigmoid(z)
    return x * (1 - x)
