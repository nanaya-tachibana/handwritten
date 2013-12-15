# -*- coding: utf-8 -*-
import numpy as np
from nylearn.utils import shared


class Dataset:

    def __init__(self, features, values):

        assert isinstance(features, np.ndarray)
        assert isinstance(values, np.ndarray)

        self.X = shared(features, dtype=features.dtype.name)
        self.y = shared(values, dtype=values.dtype.name)
