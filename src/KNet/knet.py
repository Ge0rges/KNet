"""
The lightning modules for KNet.
"""

from torch.nn import functional as functional
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

import os
import torch
import torchvision.transforms as transforms
import pytorch_lightning as pl


class KNet(pl.LightningModule):
    """
    The KNet lightning module.
    """

    def __init__(self, input_shape, layers):
        super(KNet, self).__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(layers[i], layers[i+1]) for i in range(len(layers) - 1)])
        self.layers.insert(0, torch.nn.Linear(input_shape, layers[0]))
        print(self.layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for i, l in enumerate(self.layers):
            x = l(x)
        return torch.relu(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        return {'loss': functional.cross_entropy(y_hat, y)}

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        return {'val_loss': functional.cross_entropy(y_hat, y)}

    def validation_end(self, outputs):
        # OPTIONAL
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': val_loss_mean}

    def test_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        return {'test_loss': functional.cross_entropy(y_hat, y)}

    def test_end(self, outputs):
        # OPTIONAL
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        return {'test_loss': test_loss_mean}

    def configure_optimizers(self):
        # REQUIRED
        return torch.optim.Adam(self.parameters(), lr=0.02)

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(MNIST(os.getcwd(), train=True, download=True,
                          transform=transforms.ToTensor()), batch_size=32)

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        # can also return a list of val dataloaders
        return DataLoader(MNIST(os.getcwd(), train=True, download=True,
                          transform=transforms.ToTensor()), batch_size=32)

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        # can also return a list of test dataloaders
        return DataLoader(MNIST(os.getcwd(), train=False, download=True,
                          transform=transforms.ToTensor()), batch_size=32)
