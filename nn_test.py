#! /usr/bin/env python3
# -*- coding: utf-8 -*-
from mnist.mnist import get_data
from nylearn.nnet import nnet
from nylearn.dataset import Dataset
from nylearn.train import ConjugateGradientDescent as cg
from nylearn.train import StochasticGradientDescent as sgd

tr_x, tr_y = get_data(60000, 'train')
te_x, te_y = get_data(10000, 't10k')
training_set = Dataset(tr_x, tr_y)
test_set = Dataset(te_x, te_y)

nn = nnet(tr_x.shape[1], [500], 10, 1)
s = sgd(nn)
s.train(training_set, 200, batch_size=200, alpha=30)
print('training set error: {}, test set error: {}'.format(
    nn.errors(training_set), nn.errors(test_set)))
