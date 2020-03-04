import torch.nn as nn
import torch


class AutoEncoder(nn.Module):
    def __init__(self, layer_sizes=(28*28, 128, 64, 12, 10), oldWeights=None):
        super(AutoEncoder, self).__init__()

        # Encoder
        encoder_layers = [
            self.get_layer(layer_sizes[0], layer_sizes[1], oldWeights, 0)
        ]

        for i in range(1, len(layer_sizes) - 1):
            encoder_layers.append(nn.ReLU(True))
            encoder_layers.append(self.get_layer(layer_sizes[i], layer_sizes[i + 1], oldWeights, i))

        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = [
            self.get_layer(layer_sizes[-2], layer_sizes[-3], oldWeights, len(layer_sizes) - 1)
        ]

        for i in range(len(layer_sizes) - 3, -1, -1):
            decoder_layers.append(nn.ReLU(True))
            decoder_layers.append(self.get_layer(layer_sizes[i + 1], layer_sizes[i], oldWeights, i))

        decoder_layers[-1] = nn.Tanh()

        self.decoder = nn.Sequential(*decoder_layers)

        # Activation
        self.activation = nn.Sigmoid()

        # Action
        # TODO Does OldWeights go to action?
        action_layers = [
            nn.Linear(10, 10),
            nn.ReLU(True),
            nn.Linear(10, 10),
            nn.ReLU(True),
            nn.Linear(10, 10),
            nn.Softmax()
        ]

        self.action = nn.Sequential(*action_layers)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.encoder(x)
        y = self.action(x)
        x = self.decoder(x)
        x = self.activation(x)

        return torch.cat([x, y], 1)

    def get_layer(self, input, output, weights=None, index=0):
        if weights is None:
            return nn.Linear(input, output)

        weights = weights[index]

        layer_weights = torch.zeros([output, input])
        layer = nn.Linear(input, output)

        for i in range(len(weights)):
            for j in range(len(weights[i])):
                layer_weights[i][j] = weights[i][j]

        layer.weight = nn.Parameter(layer_weights)
        return layer
