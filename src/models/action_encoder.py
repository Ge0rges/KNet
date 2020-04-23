import torch.nn as nn
import torch
import numpy as np
from torch.utils.checkpoint import checkpoint_sequential, checkpoint
from src.utils.misc import ModuleWrapperIgnores2ndArg


class ActionEncoder(nn.Module):
    def __init__(self, sizes, oldWeights=None, oldBiases=None, connected=False, ff=True):

        # Safer
        if ff:
            sizes["encoder"][-1] = sizes["action"][-1]
            sizes["action"][0] = sizes["encoder"][-1]

        # Ensure proper autoencoder size
        if "decoder" in sizes.keys():
            sizes["decoder"][0] = sizes["encoder"][-1]
            sizes["decoder"][-1] = sizes["encoder"][0]

        else:
            sizes["decoder"] = list(reversed(sizes["encoder"]))

        super(ActionEncoder, self).__init__()
        self.phase = 'ACTION'
        self.connected = connected
        self.ff = ff

        self.sizes = sizes
        self.input_size = sizes["encoder"][0]

        # Encoder
        encoder_layers = self.set_module('encoder', oldWeights=oldWeights, oldBiases=oldBiases)
        encoder_layers.append(nn.Sigmoid())  # Must be non-linear
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = self.set_module('decoder', oldWeights=oldWeights, oldBiases=oldBiases)
        decoder_layers.append(nn.Sigmoid())  # Must be non-linear
        self.decoder = nn.Sequential(*decoder_layers)

        # Action
        action_layers = self.set_module('action', oldWeights=oldWeights, oldBiases=oldBiases)
        action_layers.append(nn.Sigmoid())  # Domain [0,1] for BCELoss .
        self.action = nn.Sequential(*action_layers)

        self.dummy_tensor = torch.ones(1, requires_grad=True)

        self.action = ModuleWrapperIgnores2ndArg(self.action)
        self.encoder = ModuleWrapperIgnores2ndArg(self.encoder)
        self.decoder = ModuleWrapperIgnores2ndArg(self.decoder)

        # Make float
        self.float()

    def forward(self, x):
        checkpoints = False

        x = (checkpoint(self.encoder, x, self.dummy_tensor) if checkpoints else self.encoder(x, self.dummy_tensor))

        if self.connected:
            ci = x
        else:
            ci = torch.zeros(x.shape)
            ci.data = x.detach()

        y = (checkpoint(self.action, ci, self.dummy_tensor) if checkpoints else self.action(ci, self.dummy_tensor))

        if self.phase is 'ACTION':
            if self.ff:
                return x
            return y

        x = (checkpoint(self.decoder, x, self.dummy_tensor) if checkpoints else self.decoder(x, self.dummy_tensor))

        if self.phase is 'GENERATE':
            return x

        if self.phase is 'BOTH':
            return torch.cat([x, y], 1)

        raise ReferenceError

    def set_module(self, label, oldWeights=None, oldBiases=None):
        sizes = self.sizes[label]

        if oldWeights:
            oldWeights = oldWeights[label]
        if oldBiases:
            oldBiases = oldBiases[label]

        layers = [
            self.get_layer(sizes[0], sizes[1], oldWeights, oldBiases, 0)
        ]

        for i in range(1, len(sizes) - 1):
            layers.append(nn.LeakyReLU())
            layers.append(self.get_layer(sizes[i], sizes[i+1], oldWeights, oldBiases, i))

        return layers

    def get_layer(self, input, output, init_weights=None, init_biases=None, index=0):
        layer = nn.Linear(input, output)
        torch.nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='leaky_relu')

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
            weights = layer.weight.detach()

            if input != weights.shape[1]:
                kaiming_weights = torch.rand(weights.shape[0], input - weights.shape[1])
                torch.nn.init.kaiming_uniform_(kaiming_weights, mode='fan_in', nonlinearity='leaky_relu')

                weights = torch.cat([weights.float(), kaiming_weights.float()], dim=1)

            if output != weights.shape[0]:
                kaiming_weights = torch.rand(output - weights.shape[0], input)
                torch.nn.init.kaiming_uniform_(kaiming_weights, mode='fan_in', nonlinearity='leaky_relu')

                weights = torch.cat([weights.float(), kaiming_weights.float()], dim=0)

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
            biases = layer.bias.detach()

            if output != biases.shape[0]:
                rand_biases = torch.rand(output - biases.shape[0])
                biases = torch.cat([biases.float(), rand_biases.float()], dim=0)

            # Set
            layer.bias = nn.Parameter(biases)

            # Update the oldBiases to include padding
            init_biases[index] = layer.bias.detach()

        return layer
