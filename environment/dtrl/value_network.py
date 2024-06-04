import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ValueNetwork(nn.Module):
    def __init__(self,input_size):
        super(ValueNetwork, self).__init__()
        self.input_size= input_size
        self.net = nn.Sequential(
            nn.Linear(self.input_size, 256),
            nn.ReLU(),
            nn.Linear(256,1),
        )

    def forward(self,x):
        return self.net(x)
