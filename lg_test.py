#! /usr/bin/env python3
# -*- coding: utf-8 -*-
from mnist.mnist import get_data
from nylearn.logistic import LogisticRegression
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
lg = LogisticRegression(tr_x.shape[1], 10, 0.01)
tn = mbgd(lg)
last = tn.train(training_set, maxiter=200, batch_size=500,
                validation_set=validation_set, momentum=m, eta=0.1)
print('training set error: {}, test set error: {}'.format(
    lg.errors(training_set), lg.errors(test_set)))
lg.theta = last
print('last theta test set error: {}'.format(lg.errors(test_set)))
