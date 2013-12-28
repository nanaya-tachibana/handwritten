#! /usr/bin/env python3
# -*- coding: utf-8 -*-
from mnist.mnist import get_training_set, get_test_set
from nylearn.lenet import LeNet5
from nylearn.dataset import Dataset
from nylearn.train import ConjugateGradientDescent as cg
from nylearn.train import MinibatchGradientDescent as mbgd
from nylearn.train import momentum

tr_x, tr_y = get_training_set(6000, size=32)
te_x, te_y = get_test_set(10000, size=32)
training_set = Dataset(tr_x[:5000]/256, tr_y[:5000])
validation_set = Dataset(tr_x[5000:]/256, tr_y[5000:])
test_set = Dataset(te_x/256, te_y)

m = momentum(0.9)
convnn = LeNet5(32, [5, 50], [100], 10, batch_size=50, lamda=0.01)
tn = cg(convnn)
last = tn.train(training_set, maxiter=20)# batch_size=50, momentum=m,
#                validation_set=validation_set, eta=0.1)
print('validation set error: {}, test set error: {}'.format(
    convnn.errors(validation_set), convnn.errors(test_set)))
# convnn.theta = last
# print('last theta test set error: {}'.format(convnn.errors(test_set)))
