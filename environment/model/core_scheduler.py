import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from environment.model.DDT import DDT


class CoreScheduler(nn.Module):
    def __init__(self, devices):
        super(CoreScheduler, self).__init__()
        self.devices = devices
        self.num_features = 9
        self.forest = [self.createTree(
            device) for device in devices if device['type'] != 'cloud']
        self.optimizers = [optim.Adam(tree.parameters(), lr=0.005)
                           for tree in self.forest]

    def createTree(self, device):
        return DDT(num_input=self.num_features, num_output=device['num_cores'], depth=0, max_depth=np.log2(device['num_cores']))
