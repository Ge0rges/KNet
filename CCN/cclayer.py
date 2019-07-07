# Cascade correlation layer

import numpy as np
from utils import *

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
        """
        Sumsincoming input signals to layer, used for hidden unit activations
        """
        self.value = np.zeros(values[0].shape)

        for val in values:
            self.value = np.add(self.value, val)
        
    def compute_net_out(self):
        """
        Computes outputs to output layer of network
        """
        z  = np.dot(self.o_weights.T, self.value) + self.bias
        return sigmoid(z) 

    def compute_hidden_out(self):
        """
        Computes outputs to all hidden layers this layer is connected to
        """
        hidden_outs = []
        for w in self.h_weights:
            h = np.dot(w.T, self.value) + self.bias
            hidden_outs.append(sigmoid(h))

        return hidden_outs

    def add_hidden_weight(self, num_hidden_nodes):

        new_hidden_weight = np.random.rand(self.num_nodes, num_hidden_nodes)
        self.h_weights.append(new_hidden_weight)


    def update_output_weight(self, alpha, grad):

        self.o_weights = self.o_weights - alpha * grad

    def update_hidden_weights(self, alpha, grad, i):

        self.h_weights[i] = self.h_weights[i] - alpha * grad
