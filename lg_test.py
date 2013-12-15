#! /usr/bin/env python3
# -*- coding: utf-8 -*-
from mnist.mnist import get_data
from nylearn.logistic import LogisticRegression
from nylearn.dataset import Dataset
from nylearn.train import ConjugateGradientDescent as cg
from nylearn.train import StochasticGradientDescent as sgd

tr_x, tr_y = get_data(60000, 'train')
te_x, te_y = get_data(10000, 't10k')
training_set = Dataset(tr_x, tr_y)
test_set = Dataset(te_x, te_y)

lg = LogisticRegression(tr_x.shape[1], 10, .1)
tn = sgd(lg)
tn.train(training_set, 200, 800)
print('training set error: {}, test set error: {}'.format(
    lg.errors(training_set), lg.errors(test_set)))
