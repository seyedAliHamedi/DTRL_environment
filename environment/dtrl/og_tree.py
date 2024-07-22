import numpy as np
import torch
import torch.nn as nn


class OGTree(nn.Module):
    def __init__(self, num_input, num_output, depth, max_depth, ):
        super(OGTree, self).__init__()
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
            self.left = OGTree(num_input, num_output, depth + 1,
                               max_depth)
            self.right = OGTree(num_input, num_output, depth + 1,
                                max_depth)

    def forward(self, x):
        if self.depth == self.max_depth:
            return self.prob_dist
        val = torch.sigmoid(
            self.alpha * (torch.matmul(x, self.weights) + self.bias))

        if val >= 0.5:
            right_output = self.right(x)
            return val * right_output
        else:
            left_output = self.left(x)
            return (1 - val) * left_output
