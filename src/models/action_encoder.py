import torch.nn as nn
import torch
import numpy as np
import torch.nn.utils.prune as prune

# from torch.utils.checkpoint import checkpoint
# from src.utils.misc import ModuleWrapperIgnores2ndArg


class ActionEncoder(nn.Module):
    def __init__(self, sizes: dict, oldWeights=None, oldBiases=None, connected=False, ff=True):
        super(ActionEncoder, self).__init__()

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

        # Encoder out dictates other inputs
        sizes["action"][0] = sizes["encoder"][-1]
        sizes["decoder"][0] = sizes["encoder"][-1]

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

        # Make float
        self.float()

    def forward(self, x):
        x = self.encoder(x)

        if self.connected:
            ci = x
        else:
            ci = torch.zeros(x.shape)
            ci.data = x.clone().detach()

        y = self.action(ci)

        if self.phase == 'ACTION':
            if self.ff:
                return x
            return y

        x = self.decoder(x)

        if self.phase == 'GENERATE':
            return x

        if self.phase == 'BOTH':
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
            layers.append(self.get_layer(sizes[i], sizes[i+1], init_weights=oldWeights, init_biases=oldBiases, index=i))

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

            if input > weights.shape[1]:
                kaiming_weights = torch.rand(weights.shape[0], input - weights.shape[1]).to(weights.device)
                torch.nn.init.kaiming_uniform_(kaiming_weights, mode='fan_in', nonlinearity='leaky_relu')

                weights = torch.cat([weights.float(), kaiming_weights.float()], dim=1)

            if output > weights.shape[0]:
                kaiming_weights = torch.rand(output - weights.shape[0], input).to(weights.device)
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
                rand_biases = torch.rand(output - biases.shape[0]).to(biases.device)

                biases = torch.cat([biases.float(), rand_biases.float()], dim=0)

            # Set
            layer.bias = nn.Parameter(biases)

            # Update the oldBiases to include padding
            init_biases[index] = layer.bias.detach()

        return layer.float()

    def get_used_keys(self):
        if self.ff:
            return ["encoder", "action", "decoder"]
        else:
            if self.phase == "ACTION":
                return ["encoder", "action"]
            elif self.phase == "GENERATE":
                return ["encoder", "decoder"]
            elif self.phase == "BOTH":
                return ["encoder", "action", "decoder"]
            else:
                raise ReferenceError
