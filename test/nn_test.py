#! /usr/bin/env python3
# -*- coding: utf-8 -*-
from mnist.mnist import get_training_set, get_test_set
from nylearn.nnet import nnet
from nylearn.dataset import Dataset
from nylearn.train import ConjugateGradientDescent as cg
from nylearn.train import MinibatchGradientDescent as mbgd
from nylearn.train import momentum, decay

tr_x, tr_y = get_training_set(60000)
te_x, te_y = get_test_set(10000)
training_set = Dataset(tr_x[:50000]/256, tr_y[:50000])
validation_set = Dataset(tr_x[50000:]/256, tr_y[50000:])
test_set = Dataset(te_x/256, te_y)

m = momentum(0.9)
d = decay(0.9, 10)
nn = nnet(tr_x.shape[1], [500], 10, lamda=0.01)
tn = mbgd(nn)
last = tn.train(training_set, maxiter=1000, batch_size=50, eta=0.01,
                validation_set=validation_set, momentum=m, adjust_eta=d)
print('validation set error: {}, test set error: {}'.format(
    nn.errors(validation_set), nn.errors(test_set)))
nn.theta = last
print('last theta test set error: {}'.format(nn.errors(test_set)))
