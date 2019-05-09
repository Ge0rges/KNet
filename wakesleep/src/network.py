import numpy as np
import random


class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.

        """
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a-y) * sigmoid_prime(z)


class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).

        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.

        """
        return a-y


class WakeSleep:
    def __init__(self, sizes, cost=CrossEntropyCost):
        """
        Constructs a WakeSleep Network using sizes as a param. [2, 3, 1]
        initializes as a network with 3 layers. Starting with 2 as the input and
        output layer.
        :param sizes: Sizes of layers.
        :param cost: Cost function for the network.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.cost = cost

        # Generate the encoding layer of weights.
        biases = [np.random.randn(y, 1) for y in sizes[1:]]
        weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.encoding = {
            'b': biases,
            'w': weights
        }

        # Generate the generative layer of weights.
        biases = [np.random.randn(y, 1) for y in sizes[1:]]
        weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.gen = {
            'b': biases,
            'w': weights
        }

        self.wake = False

    def encode(self, a):
        for b, w in zip(self.encoding['b'], self.encoding['w']):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def generate(self, a):
        for b, w in zip(self.gen['b'], self.gen['w']):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def wake(self, training_data, epochs, mini_batch_size, eta):
        self.wake = True


    def sleep(self, epochs, mini_batch_size, eta,):
        self.wake = False

    def SGD(self, training_data, epochs, mini_batch_size, eta):
        n = len(training_data)

        for i in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[j: j + mini_batch_size]
                for j in xrange(0, n, mini_batch_size)
                ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

    def update_mini_batch(self, mini_batch, eta):
        biases, weights = self.get_net()

        nabla_b = [np.zeros(b.shape) for b in biases]
        nabla_w = [np.zeros(w.shape) for w in weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(weights, nabla_w)]
        biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(biases, nabla_b)]

        self.set_net(biases, weights)

    def backprop(self, x, y):
        biases, weights = self.get_net()

        nabla_b = [np.zeros(b.shape) for b in biases]
        nabla_w = [np.zeros(w.shape) for w in weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(biases, weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost.delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return nabla_b, nabla_w

    def get_net(self):
        if self.wake:
            biases = self.encoding['b']
            weights = self.encoding['w']
        else:
            biases = self.gen['b']
            weights = self.gen['w']

        return biases, weights

    def set_net(self, biases, weights):
        if self.wake:
            self.encoding['b'] = biases
            self.encoding['w'] = weights
        else:
            self.gen['b'] = biases
            self.gen['w'] = weights


def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

