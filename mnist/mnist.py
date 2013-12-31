# -*- coding: utf-8 -*-
from struct import unpack
import numpy as np


def read_images(f, row, col, num):
    pixels = row * col
    buf = f.read(pixels*num)
    return [
        unpack('B'*pixels, buf[p:p+pixels])
        for p in np.arange(0, num*pixels, pixels)
    ]


def read_labels(f, num):
    return unpack('B'*num, f.read(num))


def get_training_set(num, size=28, distortion=None):
    """Return mnist training set"""
    x = 'train'
    y = x
    if size != 28:
        x = x + str(size)
    if distortion is not None:
        x = x + '-' + distortion
    return get_mnist_samples(num, x), get_mnist_labels(num, y)


def get_test_set(num, size=28):
    """Return mnist test set"""
    x = 't10k'
    y = x
    if size != 28:
        x = x + str(size)
    return get_mnist_samples(num, x), get_mnist_labels(num, y)


def get_mnist_samples(num, t):
    return get_samples(num, 'mnist/'+t+'-images.idx3-ubyte')


def get_mnist_labels(num, t):
    return get_labels(num, 'mnist/'+t+'-labels.idx1-ubyte')


def get_samples(num, f):
    with open(f, mode='rb') as images:
        images.seek(4)
        # read total number of images and number of row pixels and colum pixels
        _num, row, col = unpack('>III', images.read(12))
        if num > _num:
            num = _num
        img = read_images(images, row, col, num)
        return np.array(img, dtype=np.uint8)


def get_labels(num, f):
    with open(f, mode='rb') as labels:
        labels.seek(4)
        _num = unpack('>I', labels.read(4))[0]
        if num > _num:
            num = _num
        l = read_labels(labels, num)
        return np.array(l, np.uint8)
