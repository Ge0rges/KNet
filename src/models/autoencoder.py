import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, layer_sizes=(28*28, 128, 64, 12, 10)):
        super(AutoEncoder, self).__init__()

        encoder_layers = []

        for i in range(len(layer_sizes) - 1):
            encoder_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            encoder_layers.append(nn.ReLU(True))
        encoder_layers = encoder_layers[:len(encoder_layers) - 1]

        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []

        for i in range(len(layer_sizes) - 2, -1, -1):
            decoder_layers.append(nn.Linear(layer_sizes[i + 1], layer_sizes[i]))
            decoder_layers.append(nn.ReLU(True))
        decoder_layers[len(decoder_layers) - 1] = nn.Tanh()

        self.decoder = nn.Sequential(*decoder_layers)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.activation(x)
        return x
