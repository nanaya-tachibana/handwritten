#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from mnist.mnist import get_training_set, get_test_set
from nylearn.lenet import LeNet
from nylearn.dataset import Dataset
from nylearn.train import ConjugateGradientDescent as cg
from nylearn.train import MinibatchGradientDescent as mbgd
from nylearn.train import momentum, decay

tr_x, tr_y = get_training_set(60000)
trd_x, trd_y = get_training_set(60000, distortion='elastic')
te_x, te_y = get_test_set(10000)
training_set = Dataset(np.vstack([tr_x[:40000], trd_x])/256,
                       np.hstack([tr_y[:40000], trd_y]))
validation_set = Dataset(tr_x[40000:]/256, tr_y[40000:])
test_set = Dataset(te_x/256, te_y)

m = momentum(0.5, 0.99, end=100)
d = decay(0.9, 10)
convnn = LeNet(28, [5, 25], [200], 10, lamda=0.01)
tn = mbgd(convnn)
last = tn.train(training_set, maxiter=150, batch_size=100, eta=0.1,
                validation_set=validation_set, momentum=m, adjust_eta=d)
print('validation set error: {}, test set error: {}'.format(
    convnn.errors(validation_set), convnn.errors(test_set)))
convnn.theta = last
print('last theta test set error: {}'.format(convnn.errors(test_set)))
