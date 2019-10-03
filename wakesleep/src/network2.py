import numpy as np
from CCN.ccnet import ccnet


class WakeSleep:
    def __init__(self, data_size, inner_size, learning_rate=1.0, network_loss_threshold=2.5):
        """
        Constructs a WakeSleep Network using sizes as a param. [2, 3, 1]
        initializes as a network with 3 layers. Starting with 2 as the input and
        output layer.
        :param sizes: Sizes of layers.
        :param cost: Cost function for the network.
        """
        self.data_size = data_size
        self.inner_size = inner_size
        self.learning_rate = learning_rate
        self.network_loss_threshold = network_loss_threshold
        self.gen_net = None
        self.encode_net = None

        self.new_gen()
        self.new_encode()

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
        self.new_gen()
        training_data = np.asarray(training_data)
        representation = np.asarray([self.encode(x)['y'][0] for x in training_data])
        self.gen_net.train(representation, training_data, epochs)

    def sleep_phase(self, epochs):
        self.new_encode()
        training_data = np.asarray([np.random.randn(self.inner_size, 1) for _ in range(1000)])
        representation = np.asarray([self.generate(x)['y'][0] for x in training_data])

        self.encode_net.train(representation, training_data, epochs)

    def new_gen(self):
        self.gen_net = ccnet(self.inner_size, self.data_size, self.learning_rate/self.inner_size, self.network_loss_threshold/self.inner_size)
        self.gen_net.create_candidate_layer()

    def new_encode(self):
        self.encode_net = ccnet(self.data_size, self.inner_size, self.learning_rate/self.data_size, self.network_loss_threshold/self.data_size)
        self.encode_net.create_candidate_layer()