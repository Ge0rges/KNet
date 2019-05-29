# Cascade correlation layer

import numpy as np
from utils import relu

class cclayer:

    def __init__(self, num_nodes, num_in, num_out):

        self.num_nodes = num_nodes
        self.num_in = num_in
        self.num_out = num_out
        self.bias = 1.0
        # connection weights to output layer
        self.o_weight = np.random.rand(num_nodes, num_out) 

        # connection weights to hidden layers
        self.h_weights = []
        self.value = None
        
    def sum_incoming_inputs(self, inputs):

        self.value = inputs[0]
        for x in inputs[1:]:
            self.value += x


    def compute_net_out(self):
        """
        Computes outputs to output layer of network
        """

        z  = np.dot(self.o_weight.T, self.value) + self.bias
        return relu(z) 

    def compute_hidden_out(self):
        """
        Computes outputs to all hidden layers this layer is connected to
        """
        hidden_outs = []
        for w in self.h_weights:
            h = np.dot(w.T, self.value) + self.bias
            hidden_outs.append(relu(h))

        return hidden_outs

    def add_hidden_weight(self, num_hidden_nodes):

        new_hidden_weight = np.random.rand(self.num_nodes, num_hidden_nodes)
        self.h_weights.append(new_hidden_weight)
