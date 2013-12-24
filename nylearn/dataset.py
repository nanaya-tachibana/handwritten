# -*- coding: utf-8 -*-
import numpy as np
import theano.tensor as T
from nylearn.utils import shared


class Dataset:

    def __init__(self, features, values):

        assert isinstance(features, np.ndarray) and features.ndim >= 1
        assert isinstance(values, np.ndarray)
        assert features.shape[0] == values.shape[0]

        self.size = values.shape[0]
        origin_X = shared(features)
        origin_y = shared(values.flatten())
        self.origin = (origin_X, origin_y)
        self.X = T.cast(origin_X, features.dtype.name)
        self.y = T.cast(origin_y, values.dtype.name)

    def slice(self, start, end):
        return Dataset(self.origin[0].get_value(borrow=True)[start:end],
                       self.origin[1].get_value(borrow=True)[start:end])

    def permute(self):
        indx = np.random.permutation(self.size)
        for v in self.origin:
            v.set_value(v.get_value(borrow=True)[indx], borrow=True)
