#! /usr/bin/env python3
# -*- coding: utf-8 -*-
from struct import pack
import numpy as np

from mnist import get_samples


def add_padding(num, size, t):
    name = '{}-images.idx3-ubyte'
    msb = 2051
    x = get_samples(num, name.format(t))
    x = x.reshape((num, 28, 28))
    px = np.zeros((num, size, size), dtype=np.uint8)
    px[:, 2:30, 2:30] = x
    px = px.reshape(num, size**2)
    filename = name.format(t+str(size))
    with open(filename, mode='xb') as f:
        f.write(pack('>IIII', msb, num, size, size))
        for i in range(num):
            f.write(pack('B'*size**2, *px[i, :]))


if __name__ == '__main__':
    add_padding(60000, 32, 'train')
    add_padding(10000, 32, 't10k')
