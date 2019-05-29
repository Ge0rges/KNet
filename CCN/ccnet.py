# Cascade correlation network

import numpy as np
from cclayer import cclayer
from utils import relu, relu_grad, l2_loss

class ccnet:

    def __init__(self, num_in, num_out, learning_rate, network_loss_threshold):

        # i/o config
        self.num_in = num_in
        self.num_out = num_out

        # hyperparameters
        self.alpha = learning_rate
        self.epsilon = network_loss_threshold
        #

        # define initial net
        self.layers = []

        input_layer = cclayer(num_in, 0, num_out)

        self.layers.append(input_layer)
        
    def forward_pass(self, x, candidate_net=False):
        """
        Computes forward pass of network with x as input vector
        Argument candidate_net specifies if we are calculating forward pass
        of the complete network, or candidate network
        """

        # compute first layer output vectors
        curr_layer = self.layers[0]
        curr_layer.sum_incoming_inputs([x])
        hidden_comps = curr_layer.compute_hidden_out()
        y = curr_layer.compute_net_out()

        if candidate_net:
            cache = None
            
        # hidden layers forward pass
        j = 1
        for i in range(1, len(self.layers)):

            curr_layer = self.layers[i]
            inputs_to_layer = hidden_comps[i - 1 : i + j]

            if candidate_net and i == len(self.layers) - 1:
                cache = inputs_to_layer

            curr_layer.sum_incoming_inputs(inputs_to_layer)
            
            hidden_comps += curr_layer.compute_hidden_out()

            if not candidate_net:
                y += curr_layer.compute_net_out()
            else:
                if i != len(self.layers) - 1:
                    y += curr_layer.compute_net_out()
            j += 1

        y_out = relu(y)

        if candidate_net:
            return y_out, cache

        return y_out, hidden_comps

    def train_network(self, x, target, epochs):
        """
        trains network connections to output
        Backpropogation is used for this training phase
        """

        n = targets.shape[0]
        # note n == len targets == len x
        m = n // epochs
        for i in range(n):
            x_data = x[i * m : (i + 1) * m]
            y_data = target[i * m : (i + 1) * m]

            print(x_data)
            print(y_data)

            # forward pass
            pred, hidden_comps = self.forward_pass(x_data)
            loss = l2_loss(pred, y_data)
            print("Loss at iter {} : {}", i, loss)
            # backward pass
            loss_grad = 1
            
            grads = []
    
        
        
        pass
    
    def train_candidate_network(self, x, targets, epochs):
        """
        trains connections to candidate layer to maximimze the correlation
        between the network error and the candidate layer activation value
        """

        n = len(x)
        for i in range(n - 1):
            x_data = x[i]
            y_data = targets[i]

            pred, cache = self.forward_pass(x_data, candidate_net=True)
            s, e_cov_sum, sign  = self.candidate_error_correlation(cache, pred, y_data)

            if i % 10 == 0:
                print("Correlation value at iter {} : {}", i, s)

            grads = []
            print(len(sign))
            print(len(self.layers))
            print(e_cov_sum)
            
            for i, layer in enumerate(self.layers[:-1]):
                input_val = cache[len(cache) - 1 - i]
                wi_grad =  sign[i] * e_cov_sum * np.dot(layer.value, relu_grad(input_val))

                # gradient ascent
                layer.h_weights[-1] += wi_grad * self.alpha
        
    def create_candidate_layer(self):
        num_nodes = 1
        num_out = self.num_out
        num_in = 0
        for layer in self.layers:
            layer.add_hidden_weight(num_nodes)
            num_in += layer.num_nodes

        new_hidden_layer = cclayer(1, num_in, num_out)
        self.layers.append(new_hidden_layer)
        
    def candidate_error_correlation(self, cache, pred, target):
        """
        computes the correlation between the candidate layer values and the
        residual output errors
        """
        avg_error = (1.0 / self.num_out) * np.sum(pred - target)
        avg_value = (1.0 / len(cache)) * np.sum(cache)

        s = 0
        # this may actually not be correct
        errors = pred - target
        e_cov = errors - avg_error
        # print(cache)
        vals = np.array(cache)
        v_cov = vals - avg_value

        print(e_cov)
        print(v_cov)
        print("~~~~~~~~~~~~~~~~~~~")
        sign = []
        
        for ec in e_cov:
            for vc in v_cov:
                p = vc * ec
                if p < 0:
                    sign.append(-1)
                else:
                    sign.append(1)
                s += p
            s = np.absolute(s)
        s = np.sum(s)

        # print(len(sign))
        return s, np.sum(e_cov), sign
    
