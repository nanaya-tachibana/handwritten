#! /usr/bin/env python3
# -*- coding: utf-8 -*-
from mnist.mnist import get_data
from nylearn.nnet import nnet
from nylearn.dataset import Dataset
from nylearn.train import ConjugateGradientDescent as cg
from nylearn.train import MinibatchGradientDescent as mbgd
from nylearn.train import momentum

tr_x, tr_y = get_data(60000, 'train')
te_x, te_y = get_data(10000, 't10k')
training_set = Dataset(tr_x[:50000], tr_y[:50000])
validation_set = Dataset(tr_x[50000:], tr_y[50000:])
test_set = Dataset(te_x, te_y)

m = momentum(0.5, 0.9, start=10, end=100)
nn = nnet(tr_x.shape[1], [tr_x.shape[1]], 10, lamda=1)
tn = mbgd(nn)
best = tn.train(training_set, maxiter=400, batch_size=500, momentum=m,
                validation_set=validation_set, eta=0.1)
print('training set error: {}, test set error: {}'.format(
    nn.errors(training_set), nn.errors(test_set)))
nn.theta = best
print('best theta test set error: {}'.format(nn.errors(test_set)))
