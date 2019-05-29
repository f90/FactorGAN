import numpy as np
import torch
import torch.nn as nn

from models.SpectralNorm import set_spectral_norm

class ConvDiscriminator(nn.Module):
    def __init__(self, x_dim, y_dim, input_channels=1, filters=32, spectral_norm=True):
        super(ConvDiscriminator, self).__init__()

        num_layers = int(np.log2(min(x_dim, y_dim)) - 3)
        feature_width = x_dim // (2 ** (num_layers+1))
        feature_height = y_dim // (2 ** (num_layers+1))

        assert(np.mod(num_layers, 1) == 0)
        num_layers = int(num_layers)

        conv_layers = list()
        conv_layers.append(set_spectral_norm(nn.Conv2d(input_channels, filters, 4, 2, 1), spectral_norm))
        conv_layers.append(nn.LeakyReLU())
        for i in range(num_layers):
            conv_layers.append(set_spectral_norm(nn.Conv2d(filters * (2 ** i), filters * (2 ** (i + 1)), 4, 2, 1), spectral_norm))
            conv_layers.append(nn.LeakyReLU())

        self.fc = set_spectral_norm(nn.Linear(feature_width * feature_height * filters * (2 ** num_layers), 1, bias=False), spectral_norm)
        self.conv_layers = nn.ModuleList(conv_layers)

    def conv_size(self, orig_size, filter_size, padding, stride):
        return np.floor((orig_size + 2*padding - filter_size).astype(float) / stride.astype(float)).astype(int) + 1

    def forward(self, input):
        x = input
        for layer in self.conv_layers:
            x = layer(x)
        x = self.fc(x.view(x.shape[0], -1))

        return torch.squeeze(x, 1)