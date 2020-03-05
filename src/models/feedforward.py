import torch.nn as nn
import torch
import numpy as np

class FeedForward(nn.Module):

    def __init__(self, sizes=(28 * 28, 312, 128, 10), oldWeights=None, oldBiases=None):
        super(FeedForward, self).__init__()
        self.input_size = sizes[0]
        self.output_size = sizes[-1]

        layers = [
            self.get_layer(sizes[0], sizes[1], oldWeights, oldBiases, 0)
        ]

        for i in range(1, len(sizes) - 1):
            layers.append(nn.ReLU(True))
            layers.append(self.get_layer(sizes[i], sizes[i + 1], oldWeights, oldBiases, i))

        layers.append(nn.Softmax())

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.classifier(x)
        return x

    def get_layer(self, input, output, weights=None, biases=None, index=0):
        if weights is None and biases is None:
            return nn.Linear(input, output)

        layer = nn.Linear(input, output)

        if weights is not None:
            weights = weights[index]

            if isinstance(weights, list):
                weights = np.asarray(weights, dtype=float)

            if isinstance(weights, np.ndarray):
                weights = torch.from_numpy(weights)

            if isinstance(weights, torch.Tensor):
                weights = nn.Parameter(weights)

            if isinstance(weights, torch.nn.Parameter):
                layer.weight = weights

        if biases is not None:
            biases = biases[index]

            if isinstance(biases, list):
                biases = np.asarray(biases, dtype=float)

            if isinstance(biases, np.ndarray):
                biases = torch.from_numpy(biases)

            if isinstance(biases, torch.Tensor):
                biases = nn.Parameter(biases)

            if isinstance(biases, torch.nn.Parameter):
                layer.weight = biases

        return layer
