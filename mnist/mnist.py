# -*- coding: utf-8 -*-
from struct import unpack
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt


def read_images(f, row, col, size):
    pixels = row * col
    buf = f.read(pixels*size)
    return [
        unpack('B'*pixels, buf[p:p+pixels])
        for p in np.arange(0, size*pixels, pixels)
    ]


def read_labels(f, size):
    return unpack('B'*size, f.read(size))


def get_data(size, t):
    images = open('mnist/'+t+'-images.idx3-ubyte', mode='rb')
    images.seek(4)
    # read total number of images and number of row pixels and colum pixels
    num, row, col = unpack('>III', images.read(12))
    if size > num:
        size = num
    img = read_images(images, row, col, size)
    images.close()

    labels = open('mnist/'+t+'-labels.idx1-ubyte', mode='rb')
    labels.seek(8)
    l = read_labels(labels, size)
    labels.close()

    return np.array(img, dtype=np.uint8), np.array(l, np.uint8)


def display_number(x):
    """Plot matrix @x as an image

    Parameters
    ------
    x: 2d array
        A pixel matrix represents a number.
    """
    plt.imshow(x, cmap=cm.Greys, vmax=x.max(), vmin=0)
    plt.show()


def display_numbers(X, size=10):
    """Plot @size x @size numbers in X randomly

    Parameters
    ------
    X: 2d array
        Each row contains one number's all pixels in row order.

    size: int
    """
    mask = np.random.permutation(X.shape[0])[0:size**2]

    display_array = np.vstack([np.hstack([
        X[mask, :][size*i+j, :].reshape(28, 28) for j in range(size)
    ]) for i in range(size)])
    plt.imshow(display_array, cmap=cm.Greys, vmax=255, vmin=0)
    plt.show()
