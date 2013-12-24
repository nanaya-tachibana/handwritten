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
training_set = Dataset(tr_x[:50000]/256, tr_y[:50000])
validation_set = Dataset(tr_x[50000:]/256, tr_y[50000:])
test_set = Dataset(te_x/256, te_y)

m = momentum(0.9)
nn = nnet(tr_x.shape[1], [784], 10, lamda=0.01)
tn = mbgd(nn)
last = tn.train(training_set, maxiter=100, batch_size=50, momentum=m,
                validation_set=validation_set, eta=0.01)
print('validation set error: {}, test set error: {}'.format(
    nn.errors(validation_set), nn.errors(test_set)))
nn.theta = last
print('last theta test set error: {}'.format(nn.errors(test_set)))
