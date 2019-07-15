# Cascade correlation network

import numpy as np
from CCN.cclayer import cclayer
from CCN.utils import *
import matplotlib.pyplot as plt

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

        Keyword arguments:
        x -- input data
        candidate_net -- If set to True, then we train the candidate net rather than the actual network
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
    
    def train_output_connections(self, x, target, epochs):
        """trains network connections to output
        Backpropogation is used for this training phase

        Keyword arguments:
        x -- input data
        target -- ground truth targets
        epochs -- number of epochs to train output and hidden connections
        """

        n = target.shape[0]
        loss = []
        # note n == len targets == len x
        for epoch in range(epochs):
            for i in range(n):
                x_data = x[i]
                y_data = target[i]

                # forward pass
                cache  = self.forward_pass(np.reshape(x_data, (x_data.shape[0], 1)))
                pred = np.zeros(cache['y'][0].shape)
                for p in cache['y']:
                    pred = np.add(pred, p)
                    
                # loss step
                loss.append(l2_loss(pred, y_data))
                # print("Loss at iter {} : {}".format(i, loss[-1]))
                
                loss_grad  = l2_loss_grad(pred, np.reshape(y_data, (y_data.shape[0], 1)))
                for i, layer in enumerate(self.layers):
                    act_grad = sigmoid_grad(np.dot(layer.o_weights.T, layer.value) + layer.bias)
                    act_grad = np.multiply(loss_grad, act_grad)
                    act_grad = np.reshape(act_grad, (act_grad.shape[0], 1))
                    o_weights_grad = np.dot(act_grad, layer.value.T).T
                    layer.update_output_weight(self.alpha, o_weights_grad)
        return loss
    
    def train_hidden_connections(self, x, targets, epochs):
        """trains hidden connections to new layer added to maximimze the correlation 
        between the network error and the candidate layer activation value

        Keyword arguments:
        x -- input data
        target -- ground truth targets
        epochs -- number of epochs to train output and hidden connections
        """
        n = targets.shape[0]
        corr = []
        for epoch in range(epochs):
            for i in range(n):
                x_data = x[i]
                y_data = targets[i]

                # forward step
                cache = self.forward_pass(np.reshape(x_data, (x_data.shape[0], 1)), candidate_net=True)
                pred = np.zeros(cache['y'][0].shape)
                for p in cache['y']:
                    pred = np.add(pred, p)

                # get correlation value
                corr_cache = self.candidate_error_correlation(cache,  np.reshape(y_data, (y_data.shape[0], 1)))
                corr.append(corr_cache['correlation'])

                # update step
                for i, layer in enumerate(self.layers[:-1]):
                    # hidden_weight = layer.h_weights[-1]
                    # x = layer.value
                    # act_grad = sigmoid_grad(np.dot(hidden_weight.T, x) + layer.bias)
                    # hidden_weight_grad = corr_cache['sign'][i] * act_grad * np.sum(corr_cache['e_cov'][i])
                    # hidden_weight_grad = np.dot(x, hidden_weight_grad)
                    # layer.update_hidden_weights(self.alpha, hidden_weight_grad)
                    pass
        return corr
        pass
    
    def create_candidate_layer(self):
        """Creates a new hidden layer to be added to the network with proper parameters and weights for
        both the hidden and output connections
        """
        num_nodes = 1
        num_out = self.num_out
        num_in = 0
        for layer in self.layers:
            layer.add_hidden_weight(num_nodes)
            num_in += layer.num_nodes

        new_hidden_layer = cclayer(1, num_in, num_out)
        self.layers.append(new_hidden_layer)
        
    def candidate_error_correlation(self, cache, target):
        """computes the correlation between the candidate layer values and the residual output errors

        Keyword arguments:
        cache -- dictionary that contains all the final output and hidden layer activations
        target -- ground truth labels
        """
        pred = np.zeros(cache['y'][0].shape)
        for p in cache['y']:
            pred = np.add(pred, p)

        hidden = cache['hidden']
        
        error = pred - target
        error_avg = np.sum(error) / len(error)
        
        activation_inputs = [hidden_acts[-1] for hidden_acts in hidden[:-1]]
        activations = []
        layer = self.layers[-1]
        for act in activation_inputs:
            activations.append(np.dot(layer.o_weights.T, act) + layer.bias)
        activations_avg = np.sum(activations) / len(activations)
 
        s = 0
        corr_cache = {'sign': [], 'e_cov': []}
        
        for j, v in enumerate(activations):
            e_cov = error - error_avg
            v_cov = v - activations_avg
            corr_cache['e_cov'].append(e_cov)

            s_true = np.sum(v_cov * e_cov)
            if s_true > 0:
                corr_cache['sign'].append(1)
            else:
                corr_cache['sign'].append(-1)
            s += abs(s_true)

        corr_cache['correlation'] = s

        return corr_cache
        pass

    def train(self, x, target, epochs):
        """
        Trains the network in following manner:
        1. Begin with a minimal net
        2. Train output connections until the loss stops decreasing
        3. If loss value is good enough, then Stop, else go to step 3
        4. Add a new hidden layer with one node
        5. Train this candidate network, till the correlation value stops decreasing
        5. Go to step 2 

        Keyword arguments:
        x -- input data
        target -- ground truth targets
        epochs -- number of epochs to train output and hidden connections
        """
        round = 0
        losses = []
        while round < 10:
            curr_loss = 0
            prev_loss = 100
            n = 0
            c = 0
            # print("Beginning : {}".format(len(self.layers)))
            # losses = self.train_output_connections(x, target, epochs)
            # loss = losses[-1]
            # print("Loss : {} \n".format(loss))
            # self.create_candidate_layer()
            # print("Middle : {}".format(len(self.layers)))
            # corr = self.train_hidden_connections(x, target, epochs)
            # print("End : {}".format(len(self.layers)))
            while abs(curr_loss - prev_loss) > self.epsilon:
                loss = np.sum(self.train_output_connections(x, target, epochs)[-1]) / 25.0
                print("Loss : {} \n".format(loss))
                prev_loss = curr_loss
                curr_loss = loss
                losses.append(loss)
            self.create_candidate_layer()

            corr = self.train_hidden_connections(x, target, epochs)
            prev_s = 0
            curr_s = corr[-1]
            k = 0
            while abs(curr_s - prev_s) > 0.01:
                corr = self.train_hidden_connections(x, target, epochs)
                
                prev_s = curr_s
                curr_s = corr[-1]
            # print("Done round {}\n".format(round))
            round += 1
        # plt.plot(losses)
        # plt.show()
        pass
