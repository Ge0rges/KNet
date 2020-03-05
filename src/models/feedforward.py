import torch.nn as nn
import torch
import numpy as np

class FeedForward(nn.Module):

    def __init__(self, sizes=(28 * 28, 312, 128, 10), oldWeights=None, oldBiases=None):
        super(FeedForward, self).__init__()
        self.input_size = sizes[0]
        self.output_size = sizes[-1]

        layers = [self.get_layer(sizes[0], sizes[1], oldWeights, oldBiases, 0)]

        for i in range(1, len(sizes) - 1):
            layers.append(nn.ReLU(True))
            layers.append(self.get_layer(sizes[i], sizes[i+1], oldWeights, oldBiases, i))

        layers.append(nn.Softmax())

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.classifier(x)
        return x

    def get_layer(self, input, output, init_weights=None, init_biases=None, index=0):
        layer = nn.Linear(input, output)

        if init_weights is not None:
            weights = init_weights[index]

            # Type checking
            if isinstance(weights, list):
                weights = np.asarray(weights, dtype=float)

            if isinstance(weights, np.ndarray):
                weights = torch.from_numpy(weights)

            if isinstance(weights, torch.Tensor):
                weights = nn.Parameter(weights)

            if isinstance(weights, torch.nn.Parameter):
                layer.weight = weights

            # Padding
            weights = layer.weight.data

            if input != weights.shape[1]:
                weights = torch.cat([weights, torch.rand(weights.shape[0], input - weights.shape[1])], dim=1)

            if output != weights.shape[0]:
                weights = torch.cat([weights, torch.rand(output - weights.shape[0], input)], dim=0)

            # Set
            layer.weight = nn.Parameter(weights)

        if init_biases is not None:
            biases = init_biases[index]

            # Type checking
            if isinstance(biases, list):
                biases = np.asarray(biases, dtype=float)

            if isinstance(biases, np.ndarray):
                biases = torch.from_numpy(biases)

            if isinstance(biases, torch.Tensor):
                biases = nn.Parameter(biases)

            if isinstance(biases, torch.nn.Parameter):
                layer.bias = biases

            # Padding
            biases = layer.bias.data
            biases = torch.cat([biases, torch.rand(output - biases.shape[0])], dim=0)

            # Set
            layer.bias = nn.Parameter(biases)

            # Update the oldBiases to include padding
            init_biases[index] = layer.bias.data

        return layer
