#! /usr/bin/env python3
# -*- coding: utf-8 -*-
from mnist.mnist import get_training_set, get_test_set
from nylearn.logistic import LogisticRegression
from nylearn.dataset import Dataset
from nylearn.train import ConjugateGradientDescent as cg
from nylearn.train import MinibatchGradientDescent as mbgd
from nylearn.train import momentum

tr_x, tr_y = get_training_set(60000)
te_x, te_y = get_test_set(10000)
training_set = Dataset(tr_x[:50000]/256, tr_y[:50000])
validation_set = Dataset(tr_x[50000:]/256, tr_y[50000:])
test_set = Dataset(te_x/256, te_y)

m = momentum(0.9)
lg = LogisticRegression(tr_x.shape[1], 10, 0.01)
tn = mbgd(lg)
last = tn.train(training_set, maxiter=200, batch_size=500, eta=0.1,
                validation_set=validation_set, momentum=m)
print('training set error: {}, test set error: {}'.format(
    lg.errors(training_set), lg.errors(test_set)))
lg.theta = last
print('last theta test set error: {}'.format(lg.errors(test_set)))
