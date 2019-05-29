import torch
import torch.nn as nn

from models.SpectralNorm import set_spectral_norm

class FCDiscriminator(nn.Module):
    def __init__(self, input_dim, spectral_norm=True, preprocess_func=None):
        '''
        Fully connected discriminator network
        :param input_dim: Number of inputs
        :param spectral_norm: Whether to use spectral normalisation
        :param preprocess_func: Function that preprocesses the input before feeding to the network
        '''
        super(FCDiscriminator, self).__init__()
        self.preprocess_func = preprocess_func

        self.fc1 = set_spectral_norm(nn.Linear(input_dim, 128), spectral_norm)
        self.fc2 = set_spectral_norm(nn.Linear(128, 128), spectral_norm)
        self.fc3 = set_spectral_norm(nn.Linear(128, 1), spectral_norm)
        self.activation = nn.LeakyReLU()
        self.output_activation = nn.Sigmoid()

    def forward(self, input):
        if self.preprocess_func != None:
            input = self.preprocess_func(input)

        x = self.fc1(input)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        return torch.squeeze(x, 1)