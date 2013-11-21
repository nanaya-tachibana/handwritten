# -*- coding: utf-8 -*-
import numpy as np

def random_parameters(n, epsilon):
    """Return a @n length vector with values in [-@epsilon,@epsilon)"""
    return np.random.random(n)*2*epsilon - epsilon

def sigmoid(z):
    """Return the sigmoid of @z

    Parameters
    ------
    z: scalar or ndarray
    the return value type depended on z
    """
    return 1.0 / (1.0 + np.exp(-z))
