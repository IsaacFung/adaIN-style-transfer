import torch
import torch.nn as nn
from util import average, stdev

class AdaIN(nn.Module):
    def __init__(self):
        super(AdaIN, self).__init__()

    def forward(self, x, y):
        x_mu = average(x)
        x_sigma = stdev(x)
        y_mu = average(y)
        y_sigma = stdev(y)

        # mu and sigma are of the shape (N, C, 1, 1) now, they can be broadcasted with (N, C, H, W)
        
        return y_sigma * ((x-x_mu) / x_sigma) + y_mu
