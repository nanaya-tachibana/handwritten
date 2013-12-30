#! /usr/bin/env python3
# -*- coding: utf-8 -*-
from struct import pack
import numpy as np

from mnist import get_samples, elastic_distortion


def distortion(num, size=28):
    msb = 2051
    t = 'train'
    if size != 28:
        t = t + str(size)
    x = get_samples(num, t)
    x = x.reshape((num, size, size))
    filename = ''.join(['mnist/train', '-elastic', '-images.idx3-ubyte'])
    with open(filename, mode='xb') as f:
        f.write(pack('>IIII', msb, num, size, size))
        for i in range(num):
            d = np.array(np.floor(elastic_distortion(x[i])), dtype=np.uint8)
            f.write(pack('B'*size**2, *d.flatten()))


if __name__ == '__main__':
    distortion(60000)
