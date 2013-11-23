# -*- coding: utf-8 -*-
import numpy as np

class Base():
    param = None
    def __init__(self, lmbd=0):
        """

        Parameters
        ------
        lmbd: float
        use for regularization
        """

        self.lmbd = lmbd
        self._cache = {'valid': False}

    @property
    def cache(self):
        if self._cache['valid']:
            return self._cache['value']
        return None

    @cache.setter
    def cache(self, value):
        self._cache['valid'] = True
        self._cache['value'] = value

    @cache.deleter
    def cache(self):
        self._cache.clear()
        self._cache['valid'] = False

    def ignore_bias(self, theta):
        _theta = theta.copy()
        if len(_theta.shape) > 1:
            _theta[:, 0] = 0
        else:
            _theta[0] = 0
        return _theta

    def add_bias(self, X):
        m = X.shape[0]
        X = np.hstack((np.ones((m,1), dtype=X.dtype), X))
        return m, X
