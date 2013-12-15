# -*- coding: utf-8 -*-
import numpy as np
import theano
import theano.tensor as T


def shared(value, name=None, dtype=''):
    """Transform @value to shared variable of type floatX.

    If @dtype is not empty, return a TensorVariable of @dtype
    refering to a TensorSharedVariable.
    """
    x = theano.shared(
        np.asarray(value, dtype=theano.config.floatX),
        name=name, borrow=True
    )
    if dtype:
        x = T.cast(x, dtype)
    return x
