# -*- coding: utf-8 -*-
from struct import unpack
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt


def read_images(f, row, col, num):
    pixels = row * col
    buf = f.read(pixels*num)
    return [
        unpack('B'*pixels, buf[p:p+pixels])
        for p in np.arange(0, num*pixels, pixels)
    ]


def read_labels(f, num):
    return unpack('B'*num, f.read(num))


def get_training_set(num, size=28):
    x = 'train'
    y = x
    if size != 28:
        x = x + str(size)
    return get_samples(num, x), get_labels(num, y)


def get_test_set(num, size=28):
    x = 't10k'
    y = x
    if size != 28:
        x = x + str(size)
    return get_samples(num, x), get_labels(num, y)


def get_samples(num, t):
    with open('mnist/'+t+'-images.idx3-ubyte', mode='rb') as images:
        images.seek(4)
        # read total number of images and number of row pixels and colum pixels
        _num, row, col = unpack('>III', images.read(12))
        if num > _num:
            num = _num
        img = read_images(images, row, col, num)
        return np.array(img, dtype=np.uint8)


def get_labels(num, t):
    with open('mnist/'+t+'-labels.idx1-ubyte', mode='rb') as labels:
        labels.seek(4)
        _num = unpack('>I', labels.read(4))[0]
        if num > _num:
            num = _num
        l = read_labels(labels, num)
        return np.array(l, np.uint8)


def display_number(x):
    """Plot matrix @x as an image

    Parameters
    ------
    x: 2d array
        A pixel matrix represents a number.
    """
    assert x.ndim == 2
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
    X = np.random.permutation(X)
    row_pixels = int(np.sqrt(X.shape[1]))

    display_array = np.vstack([np.hstack([
        X[size*i+j, :].reshape(row_pixels, row_pixels) for j in range(size)
    ]) for i in range(size)])
    plt.imshow(display_array, cmap=cm.Greys, vmax=255, vmin=0)
    plt.show()
