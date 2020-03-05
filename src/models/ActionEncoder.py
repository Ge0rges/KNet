import torch.nn as nn
import torch


class ActionEncoder(nn.Module):
    def __init__(self, encoder_sizes=(28 * 28, 128, 64, 12, 10), action_sizes=[10, 10, 10, 10], oldWeights=None):
        super(ActionEncoder, self).__init__()

        # Encoder
        encoder_layers = [
            self.get_layer(encoder_sizes[0], encoder_sizes[1], oldWeights, 0)
        ]

        for i in range(1, len(encoder_sizes) - 1):
            encoder_layers.append(nn.ReLU(True))
            encoder_layers.append(self.get_layer(encoder_sizes[i], encoder_sizes[i + 1], oldWeights, i))

        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = [
            self.get_layer(encoder_sizes[-1], encoder_sizes[-2], oldWeights, len(encoder_sizes) - 1)
        ]

        for i in range(len(encoder_sizes) - 3, -1, -1):
            decoder_layers.append(nn.ReLU(True))
            decoder_layers.append(self.get_layer(encoder_sizes[i+1], encoder_sizes[i], oldWeights, i))

        decoder_layers.append(nn.Tanh())

        self.decoder = nn.Sequential(*decoder_layers)

        # Activation
        self.activation = nn.Sigmoid()

        # Action
        action_layers = [
            self.get_layer(action_sizes[0], action_sizes[1], oldWeights, 0)
        ]

        for i in range(1, len(action_sizes) - 1):
            nn.ReLU(True)
            encoder_layers.append(self.get_layer(action_sizes[i], action_sizes[i+1], oldWeights, i))

        action_layers.append(nn.Softmax())

        self.action = nn.Sequential(*action_layers)

    def forward(self, x):
        # x = x.view(-1, 28*28)
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
