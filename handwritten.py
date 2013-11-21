#! /usr/bin/python
# -*- coding: utf-8 -*-
from struct import unpack
import numpy as np

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
    images = open(t+'-images.idx3-ubyte', mode='rb')
    images.seek(4)
    # read total number of images and number of row pixels and colum pixels
    num, row, col = unpack('>III', images.read(12))
    if size > num:
        size = num
    img = read_images(images, row, col, size)
    images.close()

    labels = open(t+'-labels.idx1-ubyte', mode='rb')
    labels.seek(8)
    l = read_labels(labels, size)

    return np.array(img, dtype=np.uint8), np.array(l, np.uint8)
