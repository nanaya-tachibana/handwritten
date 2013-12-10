#! /usr/bin/env python3
# -*- coding: utf-8 -*-
from mnist.mnist import get_data
from nylearn.logistic import LogisticRegression
from nylearn.dataset import dataset
from nylearn.train import gradient_descent_cg

tr_x, tr_y = get_data(5000, 'train')
te_x, te_y = get_data(10000, 't10k')
training_set = dataset(tr_x, tr_y)
test_set = dataset(te_x, te_y)

lg = LogisticRegression(tr_x.shape[1], 10, .1)
cg = gradient_descent_cg(lg)
cg.train(training_set, 100)
print('training set error: {}, test set error: {}'.format(
    lg.errors(training_set), lg.errors(test_set)))
