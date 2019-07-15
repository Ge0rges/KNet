# Cascade correlation layer

import numpy as np
from .utils import *

class cclayer:

    def __init__(self, num_nodes, num_in, num_out):

        self.num_nodes = num_nodes
        self.num_in = num_in
        self.num_out = num_out
        self.bias = 1.0

        self.o_weights = np.random.rand(num_nodes, num_out) 
        self.h_weights = []
        self.value = None

    def cumulate_inputs(self, values):
        """Sums incoming input signals to layer, used for hidden unit activations
        
        Keyword arguments:
        values -- all the inputs to this layer from all parent layers
        """
        self.value = np.zeros(values[0].shape)
        for val in values:
            if val.shape != self.value.shape:
                print("WTF, these shapes dont match : {} ; {}".format(self.value.shape, val.shape))
                exit()
            self.value = np.add(self.value, val)
        self.value = np.reshape(self.value, (self.value.shape[0], 1))

    def compute_net_out(self):
        """Computes outputs to output layer of network"""
        z  = sigmoid(np.dot(self.o_weights.T, self.value) + self.bias)
        return z

    def compute_hidden_out(self):
        """Computes outputs to all hidden layers this layer is connected to"""
        hidden_outs = []
        for w in self.h_weights:
            h = np.dot(w.T, self.value) + self.bias
            hidden_outs.append(sigmoid(h))

        return hidden_outs

    def add_hidden_weight(self, num_hidden_nodes):
        """Adds hidden weight to this layer

        Keyword arguments:
        num_hidden_nodes -- number of hidden nodes present in this layer
        """
        new_hidden_weight = np.random.rand(self.num_nodes, num_hidden_nodes)
        self.h_weights.append(new_hidden_weight)


    def update_output_weight(self, alpha, grad):
        """Gradient descent to update output weights

        Keyword arguments:
        alpha -- learning rate to update parameters
        grad -- gradient of hidden weight
        """
        if self.o_weights.shape != grad.shape:
            print("KYS, these shapes dont match : {} ; {}".format(self.o_weights.shape, grad.shape))
            exit()
        self.o_weights = self.o_weights - alpha * grad

    def update_hidden_weights(self, alpha, grad):
        """Gradient ascent to update hidden weights

        Keyword arguments:
        i -- index pointing to hidden weight
        alpha -- learning rate to update parameters
        grad -- gradient of hidden weight
        """
        if self.h_weights[-1].shape != grad.shape:
            print("OMG, these shapes dont match : {} ; {}".format(hidden_weight.shape, grad.shape))
            exit()
        self.h_weights[-1] = self.h_weights[-1] + alpha * grad
