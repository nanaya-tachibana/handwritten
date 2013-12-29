# -*- coding: utf-8 -*-
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

from nylearn.base import Layer
from nylearn.utils import shared, nn_random_paramters
from nylearn.logistic import LogisticRegression
from nylearn.nnet import MLP


class LeNetConvPoolLayer(Layer):
    """Pool Layer of a convolutional network"""
    def __init__(self, filter_shape, image_shape, lamda, poolsize=2):
        """
        filter_shape: tuple or list of length 4
            (number of filters, num input feature maps,
            filter height,filter width)

        image_shape: tuple or list of length 4
            (batch size or None, num input feature maps,
            image height, image width)

        poolsize: int
            the downsampling (pooling) factor
        """
        assert image_shape[1] == filter_shape[1]

        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.lamda = lamda

        self._theta = self._initialize_theta()

    def _initialize_theta(self):
        filter_shape = self.filter_shape
        image_shape = self.image_shape
        poolsize = self.poolsize

        conv_in = np.prod(filter_shape[1:])
        conv_out = filter_shape[0] * np.prod(filter_shape[2:])
        pool_out = conv_out / poolsize**2

        conv_map_size = image_shape[-1] - filter_shape[-1] + 1
        assert conv_map_size > 0
        pool_map_size = int(conv_map_size / poolsize)
        assert pool_map_size > 0

        self.conv_w = shared(
            nn_random_paramters(conv_in, conv_out, shape=filter_shape))
        self.conv_b = shared(
            nn_random_paramters(conv_in, conv_out,
                                shape=(filter_shape[0], 1, 1)))
        self.pool_w = shared(
            nn_random_paramters(conv_out, pool_out,
                                shape=(filter_shape[0], 1, 1)))
        self.pool_b = shared(
            nn_random_paramters(conv_out, pool_out,
                                shape=(filter_shape[0], 1, 1)))
        self.output_shape = (image_shape[0], filter_shape[0],
                             pool_map_size, pool_map_size)

        return [self.conv_w, self.conv_b, self.pool_w, self.pool_b]

    @property
    def theta(self):
        return np.hstack([
            x.get_value(borrow=True).flatten() for x in self._theta
        ])

    @theta.setter
    def theta(self, val):
        start = 0
        for x in self._theta:
            shape = x.shape.eval()
            end = start + np.prod(shape)
            x.set_value(val[start:end].reshape(shape), borrow=True)
            start = end

    def _gradient(self, cost):
        return T.concatenate([T.grad(cost, x).flatten() for x in self._theta])

    def _l2_regularization(self, m):
        lamda = self.lamda
        return lamda/(2*m) * (T.sum(self.conv_w**2) + T.sum(self.conv_b**2)
                              + T.sum(self.pool_w**2) + T.sum(self.pool_b**2))

    def output(self, input):
        conv_out = conv.conv2d(
            input, self.conv_w,
            filter_shape=self.filter_shape, image_shape=self.image_shape
        )
        conv_b = T.addbroadcast(self.conv_b, 1, 2)
        conv_out = conv_out + conv_b  # add bias
        pool_out = downsample.max_pool_2d(
            conv_out, (self.poolsize, self.poolsize), ignore_border=True
        )
        pool_w = T.addbroadcast(self.pool_w, 1, 2)
        pool_b = T.addbroadcast(self.pool_b, 1, 2)
        pool_out = pool_out * pool_w + pool_b

        return 1.7159*T.tanh(2/3 * pool_out)


class LeNet(Layer):
    """TODO"""

    def __init__(self, input_size, conv_layers, hidden_layers, output_size,
                 output_layer=LogisticRegression, conv_filter_size=5,
                 conv_pool_size=2, lamda=0):
        """
        Parameters:
        ------
        inpute_size: int
            The width of input image.(assume that input image is square)

        conv_layers: list
            A list of numbers of feature maps in each convlution layer.

        hidden_layers: list
            A list of numbers of units in each hidden layer
            of fully connected mlp.

        output_size: int
            The number of output layer units.

        output_layer: object

        conv_filter_size: int
            The convlution factor.

        conv_pool_size: int
            The downsampling (pooling) factor.

        lamda: float
            Parameter used for regularization. Set 0 to disable regularize.
        """
        self.lamda = shared(lamda, name='lamda')
        image_shape = (None, 1, input_size, input_size)
        self.conv_layers = []
        for c in conv_layers:
            filter_shape = (
                c, image_shape[1], conv_filter_size, conv_filter_size
            )
            cp = LeNetConvPoolLayer(
                filter_shape, image_shape, lamda, conv_pool_size)
            image_shape = cp.output_shape
            self.conv_layers.append(cp)

        self.mlp = MLP(np.prod(image_shape[1:]), hidden_layers, output_size,
                       output_layer=output_layer, lamda=self.lamda.get_value())
        self.layers = self.conv_layers + [self.mlp]

    @property
    def theta(self):
        return np.hstack([l.theta for l in self.layers])

    @theta.setter
    def theta(self, val):
        start = 0
        for l in self.layers:
            size = np.prod(l.theta.shape)
            end = start + size
            l.theta = val[start:end]
            start = end

    def _cost(self, X, y):
        return self.mlp._cost(X, y)

    def _gradient(self, cost):
        # Theano is awesome!!!
        return T.concatenate([l._gradient(cost) for l in self.layers])

    def _l2_regularization(self, m):
        return T.sum([l._l2_regularization(m) for l in self.layers])

    def _feedforward(self, X):
        input = X
        for c in self.conv_layers:
            input = c.output(input)
        return self.mlp._feedforward(input.flatten(2))

    def _cost_and_gradient(self, X, y):
        X, y = self.preprocess(X, y)
        m = y.shape[0]
        #assert m.eval() == self.batch_size

        reg = self._l2_regularization(m)
        J = self._cost(X, y) + reg
        grad = self._gradient(J)

        return J, grad

    def preprocess(self, X, y=None):
        if y is not None:
            y = y.flatten()  # make sure that y is a vector
        input_shape = list(self.layers[0].image_shape)
        input_shape[0] = -1  # batch size is not a constant
        X = X.reshape(input_shape)
        output_layer_input = self.add_bias(self._feedforward(X))
        return output_layer_input, y

    def predict(self, dataset):
        """Return predicted class labels as a numpy vecter

        Parameters
        ------
        dataset: dataset
        """
        X, _ = self.preprocess(dataset.X)
        y = theano.function([], self._predict_y(X))
        return y()

    def errors(self, dataset):
        """Return a float representing the error rate

        Parameters
        ------
        dataset: dataset
        """
        X, y = self.preprocess(dataset.X, dataset.y)
        e = theano.function([], self._errors(X, y))
        return e()

    def _predict_y(self, X):
        return self.mlp.output(X)

    def _errors(self, X, y):
        """Compute the rate of predict_y_i != y_i

        Parameters
        ------
        X: tensor like
            feature matrix

        y: tensor like
            class label
        """
        return T.mean(T.neq(self._predict_y(X), y))
