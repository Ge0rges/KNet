import numpy as np
from CCN.ccnet import ccnet


class WakeSleep:
    def __init__(self, data_size, inner_size, learning_rate=0.01, network_loss_threshold=0.5):
        """
        Constructs a WakeSleep Network using sizes as a param. [2, 3, 1]
        initializes as a network with 3 layers. Starting with 2 as the input and
        output layer.
        :param sizes: Sizes of layers.
        :param cost: Cost function for the network.
        """
        self.data_size = data_size
        self.inner_size = inner_size
        self.gen_net = ccnet(self.inner_size, self.data_size, learning_rate, network_loss_threshold)
        self.gen_net.create_candidate_layer()
        self.encode_net = ccnet(self.data_size, self.inner_size, learning_rate, network_loss_threshold)
        self.encode_net.create_candidate_layer()

    def encode(self, a):
        """Enters training data, to be encoded into the inner representation.

        :rtype: object
        """
        return self.encode_net.forward_pass(a)

    def generate(self, a):
        """Uses the inner representation/encoding to regenerate a data set.

        :rtype: object
        """
        return self.gen_net.forward_pass(a)

    def wake_phase(self, training_data, epochs):
        training_data = np.asarray(training_data)
        representation = np.asarray([self.encode(x)['y'][0] for x in training_data])
        self.gen_net.train(representation, training_data, epochs)

    def sleep_phase(self, epochs):
        training_data = np.asarray([np.random.randn(self.inner_size, 1) for _ in range(1000)])
        representation = np.asarray([self.generate(x)['y'][0] for x in training_data])

        self.encode_net.train(representation, training_data, epochs)