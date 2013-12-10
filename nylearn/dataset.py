# -*- coding: utf-8 -*-
import numpy as np
import theano
import theano.tensor as T


class dataset:

    def __init__(self, features, values):

        assert isinstance(features, np.ndarray)
        assert isinstance(values, np.ndarray)

        self.X = theano.shared(
            np.asarray(features, dtype=theano.config.floatX),
            borrow=True
        )
        self.X = T.cast(self.X, features.dtype.name)

        self.y = theano.shared(
            np.asarray(values, dtype=theano.config.floatX),
            borrow=True
        )
        self.y = T.cast(self.y, features.dtype.name)
