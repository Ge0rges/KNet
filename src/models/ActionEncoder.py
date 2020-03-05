import torch.nn as nn
import torch
import numpy as np

class ActionEncoder(nn.Module):
    def __init__(self, encoder_sizes=(28 * 28, 128, 64, 12, 10), action_sizes=[10, 10, 10, 10], oldWeights=None, oldBiases=None):
        super(ActionEncoder, self).__init__()

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

        decoder_layers.append(nn.Tanh())

        self.decoder = nn.Sequential(*decoder_layers)

        # Activation
        self.activation = nn.Sigmoid()

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
        x = self.decoder(x)
        x = self.activation(x)
        return torch.cat([x, y], 1)

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
