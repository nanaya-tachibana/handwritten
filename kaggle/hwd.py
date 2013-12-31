#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from mnist.mnist import get_training_set, get_test_set
from nylearn.lenet import LeNet
from nylearn.dataset import Dataset
from nylearn.train import MinibatchGradientDescent as mbgd
from nylearn.train import momentum, decay


def loadcsv(filename):
    return np.loadtxt(filename, delimiter=',', skiprows=1)


def distortion_set():
    return get_training_set(60000, distortion='elastic')


def mnist_test_set():
    return get_test_set(10000)


def kaggle_training_set():
    A = loadcsv('kaggle/train.csv')
    return A[:, 1:], np.array(A[:, 0], dtype=np.uint8)


def kaggle_test_set():
    return loadcsv('kaggle/test.csv')


def get_datasets():
    v = mnist_test_set()
    validation_set = Dataset(v[0]/256, v[1])

    d = distortion_set()
    ktr = kaggle_training_set()
    training_set = Dataset(np.vstack([ktr[0], d[0]])/256,
                           np.hstack([ktr[1], d[1]]))

    kte = kaggle_test_set()
    test_set = Dataset(kte[0]/256, kte[1])

    return training_set, validation_set, test_set

training_set, validation_set, test_set = get_datasets()
m = momentum(0.9, 0.99, end=100)
d = decay(0.5, 10)
convnn = LeNet(28, [5, 25], [200], 10, lamda=0.01)
tn = mbgd(convnn)
last = tn.train(training_set, maxiter=200, batch_size=100, eta=0.1,
                validation_set=validation_set, momentum=m, adjust_eta=d)
print('validation set error: {}, test set error: {}'.format(
    convnn.errors(validation_set), convnn.errors(test_set)))
convnn.save('kaggle/conv-28-5-25-200-10')
convnn.theta = last
print('last theta test set error: {}'.format(convnn.errors(test_set)))
