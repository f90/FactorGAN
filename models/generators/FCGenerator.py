import torch
import torch.nn as nn

class FCGenerator(nn.Module):
    def __init__(self, opt, output_dim):
        super(FCGenerator, self).__init__()
        self.opt = opt
        self.fc1 = nn.Linear(opt.nz, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.activation = nn.ReLU()

        if opt.generator_activation == "sigmoid":
            self.output_activation = nn.Sigmoid()
        elif opt.generator_activation == "relu":
            self.output_activation = nn.ReLU()
        else:
            print("Using generator without output activation")
            self.output_activation = None

    def forward(self, input):
        assert(len(input) == 1)

        x = self.fc1(input[0])
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        if self.output_activation != None:
            x = self.output_activation(x)
        return x