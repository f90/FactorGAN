from torch import nn as nn
from torch.nn import functional as F

class MNISTModel(nn.Module):
    def __init__(self):
        '''
        Simple CNN to use as MNIST classifier
        '''
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x, return_hidden=False):
        x = F.leaky_relu(F.avg_pool2d(self.conv1(x), 2))
        x = F.leaky_relu(F.avg_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.leaky_relu(self.fc1(x))
        x = F.dropout(x, training=self.training)

        # If return_hidden flag is given, we return the activations from the last layer instead of the final output
        if return_hidden:
            return x

        x = self.fc2(x)
        if self.training:
            return F.log_softmax(x, dim=1)
        else:
            return F.softmax(x, dim=1)