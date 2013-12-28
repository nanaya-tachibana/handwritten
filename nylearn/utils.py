# -*- coding: utf-8 -*-
import numpy as np
import theano


def shared(value, name=None):
    """Transform @value to shared variable of type floatX."""
    x = theano.shared(
        np.asarray(value, dtype=theano.config.floatX),
        name=name, borrow=True
    )
    return x


def nn_random_paramters(n_in, n_out, shape=None):
    elpsilon = np.sqrt(6 / (n_in + n_out))
    if shape is None:
        shape = (n_in, n_out)
    return np.random.uniform(-elpsilon, elpsilon, size=shape)
