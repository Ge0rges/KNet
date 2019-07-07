# Cascade correlation network

import numpy as np
from cclayer import cclayer
from utils import *

class ccnet:

    def __init__(self, num_in, num_out, learning_rate, network_loss_threshold):

        # i/o config
        self.num_in = num_in
        self.num_out = num_out

        # hyperparameters
        self.alpha = learning_rate
        self.epsilon = network_loss_threshold
        
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
        cache = {'y' : [], 'hidden' : []}
        inputs = [x]
        for i, layer in enumerate(self.layers):
            for j, h in enumerate(cache['hidden']):
                if i - j  - 1 < len(h):
                    inputs.append(h[i - j - 1])

            layer.cumulate_inputs(inputs)
            y = layer.compute_net_out()
            h = layer.compute_hidden_out()

            cache['hidden'].append(h)
            cache['y'].append(y)
            
            del inputs[:]

        if candidate_net:
            cache['y'] = cache['y'][:-1]
            
        return cache

    
    def train_network(self, x, target, epochs):
        """
        trains network connections to output
        Backpropogation is used for this training phase
        """

        n = target.shape[0]
        loss = []
        # note n == len targets == len x
        for epoch in range(epochs):
            for i in range(n):
                x_data = x[i]
                y_data = target[i]
            
                print(x_data)
                print(y_data)
            
                # forward pass
                cache  = self.forward_pass(x_data)
                pred = cache['y']

                # loss step
                loss.append(l2_loss(pred, y_data))
                print("Loss at iter {} : {}".format(i, loss[-1]))
            
                # backward pass
                # im not saure if this is actually needed
                if loss[-1] < 0:
                    loss_grad = -1
                else:
                    loss_grad = 1
                    
                loss_grad *= np.sum(l2_loss_grad(pred, y_data))
                for i, layer in enumerate(self.layers):
                    layer_val = np.reshape(layer.value, (layer.value.shape[0], 1))
                    act_grad = sigmoid_grad(np.dot(layer.o_weights.T, layer_val) + layer.bias)
                    act_grad = np.reshape(act_grad, (act_grad.shape[0], 1))
                    weights_grad = np.dot(layer_val, np.multiply(loss_grad, act_grad).T)
                    layer.update_output_weight(self.alpha, weights_grad)
        return loss
    
    def train_candidate_network(self, x, targets, epochs):
        """
        trains connections to candidate layer to maximimze the correlation
        between the network error and the candidate layer activation value
        """
        n = target.shape[0]
        corr = []
        for epoch in range(epochs):
            for i in range(n):
                x_data = x[i]
                y_data = targets[i]

                # forward step
                cache = self.forward_pass(x_i, candidate_net=True)
                pred = cache['y']

                # get correlation value
                corr_cache = self.candidate_error_correlation(cache, target)
                corr.append(corr_cache['correlation'])

                # update step
                for i, layer in enumerate(self.layers):
                    layer_val = np.reshape(layer.value, (layer.value.shape[0], 1))
                    act_grad = sigmoid_grad(cache['hidden'][i][-1])
                    act_grad = np.reshape(act_grad, (act_grad.shape[0], 1))
                    weights_grad = corr_cache['sign'][i] * np.sum(corr_cache['e_cov'] * np.dot(layer_val, act_grad.T))
                    layer.update_hidden_weights(self.alpha, weights_grad, i)
                    pass
        pass
    
    def create_candidate_layer(self):
        num_nodes = 1
        num_out = self.num_out
        num_in = 0
        for layer in self.layers:
            layer.add_hidden_weight(num_nodes)
            num_in += layer.num_nodes

        new_hidden_layer = cclayer(1, num_in, num_out)
        self.layers.append(new_hidden_layer)
        
    def candidate_error_correlation(self, cache, target):
        """
        computes the correlation between the candidate layer values and the
        residual output errors
        """

        error = cache['y'] - target[:-1]
        error_avg = np.sum(error) / len(error)

        activation_inputs = []
        for hidden_acts in cache['hidden']:
            activation_inputs.append(hidden_acts[-1])

        activations = []
        layer = self.layers[-1]
        for act in activation_inputs:
            activations.append(np.dot(layer.o_weights.T, act) + layer.bias)
        activations_avg = np.sum(activations) / len(activations)

        s = 0
        corr_cache = {'sign': [], 'e_cov': []}
        for i, o in enumerate(cache['y']):
            for j, v in enumerate(activations):
                e_cov = error[i] - error_avg
                v_cov = v - activations_avg
                corr_cache['e_cov'].append(e_cov)

                s_true = v_cov * e_cov
                if s_true > 0:
                    corr_cache['sign'].append(1)
                else:
                    corr_cache['sign'].append(-1)
                s += abs(s_true)

        corr_cache['correlation'] = s

        return corr_cache
        pass
