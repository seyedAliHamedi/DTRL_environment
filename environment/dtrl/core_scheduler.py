import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


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
        return DDTNode(num_input=self.num_features, num_output=device['num_cores'], depth=0, max_depth=np.log2(device['num_cores']))


class DDTNode(nn.Module):
    def __init__(self, num_input, num_output, depth, max_depth):
        super(DDTNode, self).__init__()
        self.depth = depth
        self.max_depth = max_depth

        if depth != max_depth:
            self.weights = nn.Parameter(torch.empty(
                num_input).normal_(mean=0, std=0.1))
            self.bias = nn.Parameter(torch.zeros(1))
            self.alpha = nn.Parameter(torch.zeros(1))
        if depth == max_depth:
            self.prob_dist = nn.Parameter(torch.zeros(num_output))
        if depth < max_depth:
            self.left = DDTNode(num_input, num_output, depth + 1, max_depth)
            self.right = DDTNode(num_input, num_output, depth + 1, max_depth)

    def forward(self, x):
        if self.depth == self.max_depth:
            return self.prob_dist
        val = torch.sigmoid(
            self.alpha * (torch.matmul(x, self.weights) + self.bias))

        if val >= 0.5:
            return val * self.right(x)
        else:
            return (1 - val) * self.left(x)
