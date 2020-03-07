import torch.nn as nn
import torch
import numpy as np


class ActionEncoder(nn.Module):
    def __init__(self, encoder_sizes=(28 * 28, 128, 64, 12, 10), action_sizes=(10, 10, 10, 10), oldWeights=None, oldBiases=None):
        super(ActionEncoder, self).__init__()
        self.phase = 'ACTION'

        # Encoder
        encoder_layers = [
            self.get_layer(encoder_sizes[0], encoder_sizes[1], oldWeights, oldBiases, 0)
        ]

        for i in range(1, len(encoder_sizes) - 1):
            encoder_layers.append(nn.ReLU(True))
            encoder_layers.append(self.get_layer(encoder_sizes[i], encoder_sizes[i + 1], oldWeights, oldBiases, i))

        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = [
            self.get_layer(encoder_sizes[-1], encoder_sizes[-2], oldWeights, oldBiases, len(encoder_sizes) - 1)
        ]

        for i in range(len(encoder_sizes) - 3, -1, -1):
            decoder_layers.append(nn.ReLU(True))
            decoder_layers.append(self.get_layer(encoder_sizes[i + 1], encoder_sizes[i], oldWeights, oldBiases, i))

        decoder_layers.append(nn.Sigmoid)

        self.decoder = nn.Sequential(*decoder_layers)

        # Action
        action_layers = [
            self.get_layer(action_sizes[0], action_sizes[1], oldWeights, oldBiases, 0)
        ]

        for i in range(1, len(action_sizes) - 1):
            nn.ReLU(True)
            encoder_layers.append(self.get_layer(action_sizes[i], action_sizes[i+1], oldWeights, oldBiases, i))

        action_layers.append(nn.Softmax())

        self.action = nn.Sequential(*action_layers)

    def forward(self, x):
        # x = x.view(-1, 28*28)
        x = self.encoder(x)
        y = self.action(x)

        if self.action is 'ACTION':
            return y
        x = self.decoder(x)
        if self.action is 'GENERATE':
            return x
        return torch.cat([x, y], 1)

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
