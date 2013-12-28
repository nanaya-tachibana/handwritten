#! /usr/bin/env python3
# -*- coding: utf-8 -*-
from nylearn.debug.checkgrad import check_classifier_grad
from nylearn.lenet import LeNet5

if __name__ == '__main__':
    m = 10
    lamda = 0.1
    input_size = 16
    output_labels = 3
    convnn = LeNet5(16, [2, 3], [2], 3, m, lamda=lamda)
    check_classifier_grad(input_size=input_size**2,
                          output_labels=output_labels, m=m,
                          model=convnn, x0=convnn.theta)
