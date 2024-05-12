import numpy as np
import torch
import  torch.nn as nn
import torch.optim as optim

class CoreScheduler(nn.Module):
    def __init__(self, devices):
        super(CoreScheduler, self).__init__()
        self.devices = devices
        self.num_features = 9
        self.exploration_factor = 0.10
        self.forest = [DDTNode(self.num_features, device['num_cores'], 0, np.log2(device['num_cores']),self.exploration_factor) for device in devices]
        self.optimizers = [optim.Adam(tree.parameters(), lr=0.01) for tree in self.forest]

    def forward(self,x,device_index):
        return self.forest[device_index](x)


class DDTNode(nn.Module):
    def __init__(
        self, num_input, num_output, depth, max_depth, tree_exploration_facotr
    ):
        super(DDTNode, self).__init__()
        self.depth = depth
        self.max_depth = max_depth
        self.tree_exploration_facotr = tree_exploration_facotr
        self.epsilon = 1e-9
        if depth != max_depth:
            self.weights = nn.Parameter(torch.zeros(num_input))
            self.bias = nn.Parameter(torch.zeros(1))
            self.alpha = nn.Parameter(torch.zeros(1))
        if depth == max_depth:
            self.prob_dist = nn.Parameter(torch.zeros(num_output))

        if depth < max_depth:
            self.left = DDTNode(num_input, num_output, depth + 1, max_depth,tree_exploration_facotr)
            self.right = DDTNode(num_input, num_output, depth + 1, max_depth,tree_exploration_facotr)

    def forward(self, x):
        if self.depth == self.max_depth:
            return self.prob_dist
        print("------")
        print(x)
        print(self.weights.t())
        print("------")



        val = torch.sigmoid(self.alpha * (torch.matmul(torch.tensor(x), self.weights.t()) + self.bias))
        a = np.random.uniform(0, 1)
        self.tree_exploration_facotr -= self.epsilon
        if a < self.tree_exploration_facotr:
            val = 1 - val
        if val >= 0.5:
            return val * self.right(x)
        else:
            return (1 - val) * self.left(x)
