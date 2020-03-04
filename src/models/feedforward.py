import torch.nn as nn
import torch


class FeedForward(nn.Module):

    def __init__(self, sizes=(28 * 28, 312, 128, 10), oldWeights=None):
        super(FeedForward, self).__init__()
        self.input_size = sizes[0]
        self.output_size = sizes[-1]

        layers = [
            self.get_layer(sizes[0], sizes[1], oldWeights, 0)
        ]

        for i in range(1, len(sizes) - 1):
            layers.append(nn.ReLU(True))
            layers.append(self.get_layer(sizes[i], sizes[i + 1], oldWeights, i))

        layers.append(nn.Softmax())

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.classifier(x)
        return x

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
