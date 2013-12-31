# -*- coding: utf-8 -*-
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.linalg import norm
import matplotlib.cm as cm
import matplotlib.pyplot as plt


def display_number(x):
    """Plot one digit.

    Parameters
    ------
    x: 2d array
        A pixel matrix represents a number.
    """
    assert x.ndim == 2
    plt.imshow(x, cmap=cm.Greys, vmax=x.max(), vmin=0)
    plt.show()


def display_numbers(X, shape=(28, 28)):
    """Plot digits.

    Parameters
    ------
    X: 2d array
        Each row contains one number's all pixels in row order.

    shape: tuple
        The shape of each digit image.
    """
    assert X.ndim == 2
    assert np.prod(shape) == X.shape[1]
    n = X.shape[0]
    row = int(np.sqrt(n)) + 1
    col = row
    filling = row*col - n  # blank filling
    X = np.vstack([X, np.zeros((filling, np.prod(shape)))])

    display_array = np.vstack([np.hstack([
        X[col*i+j, :].reshape(shape) for j in range(col)
    ]) for i in range(row)])
    plt.imshow(display_array, cmap=cm.Greys, vmax=255, vmin=0)
    plt.show()


def elastic_distortion(image, sigma=6, alpha=36):
    """The generation of the elastic distortion.

    First, random displacement fields are created from a uniform distribution
    between −1 and +1. They are then convolved with a Gaussian of standard
    deviation sigma. After normalization and multiplication by a scaling
    factor alpha that controls the intensity of the deformation, they are
    applied on the image. sigma stands for the elastic coefficient. A small
    sigma means more elastic distortion. For a large sigma, the deformation
    approaches affine, and if sigma is very large, then the displacements
    become translations.
    """
    def delta():
        d = gaussian_filter(np.random.uniform(-1, 1, size=image.shape), sigma)
        return (d / norm(d)) * alpha

    assert image.ndim == 2
    dx = delta()
    dy = delta()
    return bilinear_interpolate(image, dx, dy)


def bilinear_interpolate(values, dx, dy):
    """Interpolating with given dx and dy"""
    assert values.shape == dx.shape == dy.shape

    A = np.zeros(values.shape)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            x = i + dx[i, j]
            y = j + dy[i, j]
            if x < 0:
                x = x + int(1 + 0 - x)
            if x >= values.shape[0] - 1:
                x = x - int(1 + x - (values.shape[0] - 1))
            if y < 0:
                y = y + int(1 + 0 - y)
            if y >= values.shape[1] - 1:
                y = y - int(1 + y - (values.shape[1] - 1))

            x1 = int(x)
            y1 = int(y)
            x2 = x1 + 1
            y2 = y1 + 1
            f11 = values[x1, y1]
            f12 = values[x1, y2]
            f21 = values[x2, y1]
            f22 = values[x2, y2]

            A[i, j] = (
                f11*(x2-x)*(y2-y) + f12*(x2-x)*(y-y1)
                + f21*(x-x1)*(y2-y) + f22*(x-x1)*(y-y1)
            )
    return A
