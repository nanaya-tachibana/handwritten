# -*- coding: utf-8 -*-
import operator
from functools import reduce
import numpy as np
from scipy import linalg

from .base import Base
from .utils import sigmoid, sigmoid_gradient, random_parameters
from .optimize import mini


class lenet4(Base):

    def _neighbor_features(self, X, (i0, j0), nrange):
        """"""
        n = int(np.sqrt(X.shape[1]))
        i = np.arange(nrange)
        j = np.arange(nrange)
        indx = ((i0 + i) * n)[:, np.newaxis] + (j0 + j)
        return X[:, indx.reshape(-1)]

    def _feature_map(self, X, maps=4, nrange=5):
        """Generate @maps feature maps using input @X.

        Each unit in each output feature map is connected to @nrange by @nrange
        neighborhood in the input @X. Each unit in each output feature map
        generated by using this @nrange by @nrange features matrix to multiply
        theta.

        Parameter
        ------
        X: 2d array
        Input samples or map

        maps: integer
        Number of output feature maps

        nrange: integer
        Neighborhood should be within this distance
        """
        input_map_size = int(np.sqrt(X.shape[1]))
        output_map_size = input_map_size - nrange + 1
        input_size = (nrange**2 + 1) * output_map_size**2
        output_size = maps

        theta = random_parameters(
            (output_size, nrange**2 + 1),
            np.sqrt(6) / np.sqrt(input_size + output_size)
        )

        output = np.zeros((maps, output_map_size**2))
        for i in range(output_map_size):
            for j in range(output_map_size):
                indx = i * output_map_size + j
                output[:, indx] = self.add_bias(self._neighbor_features(
                    X, (i, j), nrange
                )).dot(theta.T)

        return output, output_map_size

    def _subsampling(self, X, maps=2, nrange=2):
        """Sub sampling @maps feature maps using input @X.

        Each unit in each output feature map is connected to @nrange by @nrange
        neighborhood in the input @X. Each unit in each output feature map
        generated by adding up this @nrange by @nrange feature matrix and
        multiply the sum and theta.
        """
        input_map_size = int(np.sqrt(X.shape[1]))
        output_map_size = int(np.floor(input_map_size / 2))
        input_size = 2 * output**2
        output_size = maps

        theta = random_parameters(
            (output_size, 2),
            np.sqrt(6) / np.sqrt(input_size + output_size)
        )

        output = np.zeros((maps, output_map_size**2))
        for i in range(output_map_size):
            for j in range(output_map_size):
                indx = i * output_map_size + j
                output[:, indx] = self.add_bias(self._neighbor_features(
                    X, (i*2, j*2), nrange
                ).sum(axis=1)).dot(theta.T)