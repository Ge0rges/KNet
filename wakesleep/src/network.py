import numpy as np

class WakeSleep():
    def __init__(self, sizes):
        """
        Constructs a WakeSleep Network using sizes as a param. [2, 3, 1]
        initializes as a network with 3 layers. Starting with 2 as the input and
        output layer.
        :param sizes: Sizes of layers.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes

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

    def encode(self, a):
        for b, w in zip(self.encoding['b'], self.encoding['w']):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def generate(self, a):
        a = self.encode(a)
        for b, w in zip(self.gen['b'], self.gen['w']):
            a = sigmoid(np.dot(w, a) + b)
        return a


def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

