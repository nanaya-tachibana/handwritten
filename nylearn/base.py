# -*- coding: utf-8 -*-
import theano.tensor as T

from nylearn.utils import shared, nn_random_paramters


class Model:
    """A class representing a model."""

    @property
    def theta(self):
        """Return the value of model's trainable weights."""
        raise NotImplementedError(str(type(self))
                                  + 'does not implement get theta')

    @theta.setter
    def theta(self, val):
        """Set the value of model's trainable weights."""
        raise NotImplementedError(str(type(self))
                                  + 'does not implement set theta')

    def _initialize_theta(self):
        """Return a set value which can be used to initialize weights."""
        raise NotImplementedError(str(type(self))
                                  + 'does not implement _initialize_theta')

    def _cost(self, X, y):
        """Return the cost of model's weights."""
        raise NotImplementedError(str(type(self))
                                  + 'does not implement _cost')

    def _gradient(self, cost):
        """Return the gradient of @cost w.r.t model's weights."""
        raise NotImplementedError(str(type(self))
                                  + 'does not implement _gradient')

    def _l2_regularization(self, m):
        """Return l2 regularization of model's weights."""
        raise NotImplementedError(str(type(self))
                                  + 'does not implement _l2_regularization')

    def predict(self, dataset):
        """Return predicted value."""
        raise NotImplementedError(str(type(self))
                                  + 'does not implement predict')

    def errors(self, dataset):
        """Return model predicting error rate."""
        raise NotImplementedError(str(type(self))
                                  + 'does not implement errors')

    def save(self, save_as):
        """Save the value of all parameters that define this model.

        Parameters
        ------
        save_as: string
            filename
        """
        raise NotImplementedError(str(type(self)) + 'does not implement save')

    def load(self, load_from):
        """Load parameters's value from file.

        Parameters
        ------
        save_as: string
            filename
        """
        raise NotImplementedError(str(type(self)) + 'does not implement load')


class Layer(Model):

    def __init__(self, n_in, n_out, lamda):
        """
        n_in: int
            Number of input units, the dimension of the space
            in which the input lie.

        n_out: int
            Number of output units, the dimension of the space
            in which the output lie.

        lamda: theano like
            Parameter used for regularization. Set 0 to disable regularize.
        """
        self.n_in = n_in
        self.n_out = n_out
        self.lamda = lamda

        self._theta = shared(self._initialize_theta())
        self.w = self._theta[1:, :]
        self.b = self._theta[0, :]

    @property
    def theta(self):
        """Return the value of model's trainable weights."""
        return self._theta.get_value(borrow=True).flatten()

    @theta.setter
    def theta(self, val):
        """Set the value of model's trainable weights."""
        self._theta.set_value(
            val.reshape(self.n_in+1, self.n_out), borrow=True)

    def _initialize_theta(self):
        """Return a set value which can be used to initialize weights.

        Override this function to use other initial value.
        """
        return nn_random_paramters(self.n_in+1, self.n_out)

    def _gradient(self, cost):
        """Return the gradient of @cost w.r.t model's weights."""
        return T.grad(cost, self._theta).flatten()

    def _l2_regularization(self, m):
        """Return l2 regularization of model's weights."""
        w = self.w
        lamda = self.lamda

        return lamda/(2*m) * T.sum(w**2)
        #reg_grad = T.concatenate([T.zeros((1, w.shape[1])), lamda * w / m])

    def output(self, input):
        """Return layer output."""
        raise NotImplementedError(str(type(self))
                                  + 'does not implement output')

    @classmethod
    def add_bias(cls, X):
        return T.concatenate([T.ones((X.shape[0], 1)), X], axis=1)
