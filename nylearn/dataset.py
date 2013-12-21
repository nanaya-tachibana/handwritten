# -*- coding: utf-8 -*-
import numpy as np
from nylearn.utils import shared


class Dataset:

    def __init__(self, features, values):

        assert isinstance(features, np.ndarray) and features.ndim >= 1
        assert isinstance(values, np.ndarray)

        self.origin = (features, values)
        self.size = values.shape[0]
        self.X = shared(features, dtype=features.dtype.name)
        self.y = shared(values.flatten(), dtype=values.dtype.name)

    def slice(self, start, end):
        return Dataset(self.origin[0][start:end], self.origin[1][start:end])

    def permute(self):
        map(np.random.permutation, self.origin)
