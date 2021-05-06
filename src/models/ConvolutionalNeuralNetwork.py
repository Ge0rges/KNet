"""
A simple example of a Convolutional Neural Network
used for reference. Source: https://www.tutorialspoint.com/pytorch/pytorch_convolutional_neural_network.htm
"""

import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalNN(nn.Module):
   def __init__(self):
        super(ConvolutionalNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(18 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, 10)

   def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1, 18 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return (x)
