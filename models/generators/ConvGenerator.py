import torch.nn as nn
import numpy as np

class ConvGenerator(nn.Module):
    def __init__(self, opt, ngf, out_width, nc=3):
        super(ConvGenerator, self).__init__()
        self.conv_size = 4

        num_layers = np.log2(out_width) - 3
        assert(np.mod(num_layers, 1) == 0)
        num_layers = int(num_layers)

        conv_list = list()

        # Compute channel numbers in each layer
        channel_list = [ngf * (2 ** (i-1)) for i in range(num_layers+1, 0, -1)]

        # First layer
        conv_list.append(nn.ConvTranspose2d(opt.nz, channel_list[0], 4, 1, 0, bias=True))
        conv_list.append(nn.ReLU(True))

        for i in range(0, num_layers):
            conv_list.append(nn.ConvTranspose2d(channel_list[i], channel_list[i+1], self.conv_size, 2, 1, bias=True))
            conv_list.append(nn.ReLU(True))

        # Last layer
        conv_list.append(nn.ConvTranspose2d(ngf, nc, self.conv_size, 2, 1, bias=True))
        if opt.generator_activation == "sigmoid":
            conv_list.append(nn.Sigmoid())
        elif opt.generator_activation == "relu":
            conv_list.append(nn.ReLU())
        else:
            print("WARNING: Using ConvGenerator without output activation")

        self.main = nn.Sequential(*conv_list)

    def forward(self, input):
        assert (len(input) == 1)
        noise = input[0].unsqueeze(2).unsqueeze(2)
        output = self.main(noise)
        return output
