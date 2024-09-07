import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from sklearn.cluster import KMeans
import torch.optim as optim

class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions,devices):
        super(ActorCritic, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.actor = DeviceDDT(devices=devices,depth=0,max_depth=3)
        self.critic = nn.Sequential(
            nn.Linear (5 + 4 * len(devices), 128), nn.ReLU(), nn.Linear(128, 1))

        self.rewards = []
        self.actions = []
        self.states = []

    def archive(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear_memory(self):
        self.rewards = []
        self.actions = []
        self.states = []

    def forward(self, x):
        p,path = self.actor(x)
        v = self.critic(x)
        return p,path, v

    def choose_action(self, ob):
        state = torch.tensor([ob], dtype=torch.float)
        pi,path, _ = self.forward(state)

        # Ensure numerical stability for softmax
        pi = pi - pi.max()
        probs = F.softmax(pi, dim=-1)

        dist = Categorical(probs)
        action = dist.sample()

        return action.item(),path

    def calculate_returns(self):
        G = 0
        gamma = 0
        returns = []
        for reward in self.rewards[::-1]:
            G = G * gamma + reward
            returns.append(G)

        returns.reverse()
        returns = torch.tensor(returns, dtype=torch.float)
        return returns

    def calc_loss(self):
        states = torch.tensor(self.states, dtype=torch.float)
        # Ensure actions are long type for indexing
        actions = torch.tensor(self.actions, dtype=torch.long)
        returns = self.calculate_returns()

        pis = []
        values = []
        for state in states:
            pi, _,value = self.forward(state)
            pis.append(pi)
            values.append(value)
        pis = torch.stack(pis, dim=0)
        values = torch.stack(values, dim=0).squeeze()

        # Ensure numerical stability for softmax
        pis = pis - pis.max(dim=-1, keepdim=True)[0]
        probs = F.softmax(pis, dim=-1)

        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        # actor_loss = -log_probs * (returns - values)
        actor_loss = -torch.sum(log_probs * returns)
        critic_loss = F.mse_loss(values, returns, reduction='none')

        # total_loss = (actor_loss + critic_loss).mean()
        total_loss = actor_loss
        return total_loss


class CoreScheduler(nn.Module):
    def __init__(self, devices):
        super(CoreScheduler, self).__init__()
        self.devices = devices
        self.num_features = 9
        self.forest = [self.createTree(device) for device in devices if device['type'] != 'cloud']
        self.optimizers = [optim.Adam(tree.parameters(), lr=0.005)for tree in self.forest]

    def createTree(self, device):
        return DDT(num_input=self.num_features, num_output=device['num_cores']*3, depth=0, max_depth=np.log2(device['num_cores']))
    
    




class DDT(nn.Module):
    def __init__(self, num_input, num_output, depth, max_depth, ):
        super(DDT, self).__init__()
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
            self.left = DDT(num_input, num_output, depth + 1,
                            max_depth)
            self.right = DDT(num_input, num_output, depth + 1,
                             max_depth)

    def forward(self, x,path=""):
        if self.depth == self.max_depth:
            return self.prob_dist, path
        val = torch.sigmoid(
            self.alpha * (torch.matmul(x, self.weights) + self.bias))

        if val >= 0.5:
            right_output, right_path = self.right(x, path + "R")
            return val * right_output, right_path
        else:
            left_output, left_path = self.left(x, path + "L")
            return (1 - val) * left_output, left_path

class DeviceDDT(nn.Module):
    def __init__(self, devices, depth, max_depth ):
        super(DeviceDDT, self).__init__()
        self.depth = depth
        self.max_depth = max_depth

        if depth != max_depth:
            self.weights = nn.Parameter(torch.empty(5+4*len(devices)).normal_(mean=0, std=0.1))
            self.bias = nn.Parameter(torch.zeros(1))
            self.alpha = nn.Parameter(torch.zeros(1))
        if depth == max_depth:
            self.prob_dist = nn.Parameter(torch.zeros(len(devices)))
        if depth < max_depth:
            self.left = DeviceDDT(devices, depth + 1,max_depth)
            self.right = DeviceDDT(devices, depth + 1,max_depth)

    def forward(self, x,path=""):
        if self.depth == self.max_depth:
            return self.prob_dist, path
        val = torch.sigmoid(
            self.alpha * (torch.matmul(x, self.weights) + self.bias))

        if val >= 0.5:
            right_output, right_path = self.right(x, path + "R")
            return val * right_output, right_path
        else:
            left_output, left_path = self.left(x, path + "L")
            return (1 - val) * left_output, left_path



# class ClusterTree(nn.Module):
#     def __init__(self, devices, depth, max_depth):
#         super(ClusterTree, self).__init__()
#         self.depth = depth
#         self.max_depth = max_depth
        
        
#         self.devices = devices
#         num_features = 5 + 4 * len(devices)
#         self.max_depth = max_depth

#         if depth != max_depth:
#             self.weights = nn.Parameter(torch.empty(
#                 num_features).normal_(mean=0, std=0.1))
#             self.bias = nn.Parameter(torch.zeros(1))
#             self.alpha = nn.Parameter(torch.zeros(1))
#         if depth == max_depth:
#             self.prob_dist = nn.Parameter(torch.zeros(len(devices)))

#         if depth < max_depth:
#             clusters = self.cluster(self.devices)
#             left_cluster = clusters[0]
#             right_cluster = clusters[1]
#             self.left = ClusterTree(left_cluster, depth+1, max_depth)
#             self.right = ClusterTree(right_cluster, depth+1, max_depth)

#     def forward(self, x, path=""):
#         if self.depth == self.max_depth:
#             return self.prob_dist,path

#         val = torch.sigmoid(
#             self.alpha * (torch.matmul(x, self.weights.t()) + self.bias))

#         if val >= 0.5:
#             indices = [self.devices.index(device)
#                        for device in self.right.devices]
#             temp = x[5:].view(-1, 4)
#             indices_tensor = torch.tensor(indices)
#             x = torch.cat((x[0:5], temp[indices_tensor].view(-1)), dim=0)
#             right_output, right_path  = self.right(x, path + "R")
#             return val * right_output,right_path
        
#         else:
#             indices = [self.devices.index(device)
#                        for device in self.left.devices]
#             temp = x[5:].view(-1, 4)
#             indices_tensor = torch.tensor(indices)
#             x = torch.cat((x[0:5], temp[indices_tensor].view(-1)), dim=0)
#             left_output, left_path = self.left(x, path + "L")
#             return (1 - val) * left_output, left_path

#     def get_devices(self, x):
#         if self.depth == self.max_depth:
#             return self.devices

#         val = torch.sigmoid(
#             self.alpha * (torch.matmul(x, self.weights.t()) + self.bias))

#         if val >= 0.5:
#             indices = [self.devices.index(device)
#                        for device in self.right.devices]
#             temp = x[5:].view(-1, 4)
#             indices_tensor = torch.tensor(indices)
#             x = torch.cat((x[0:5], temp[indices_tensor].view(-1)), dim=0)
#             return self.right.get_devices(x)
#         else:
#             indices = [self.devices.index(device)
#                        for device in self.left.devices]
#             temp = x[5:].view(-1, 4)
#             indices_tensor = torch.tensor(indices)
#             x = torch.cat((x[0:5], temp[indices_tensor].view(-1)), dim=0)
#             return self.left.get_devices(x)
#     def cluster(self, devices, k=2):
#         data = [self.get_pe_data(device) for device in devices]
#         if len(devices) < k:
#             return [devices]*k
#         X = np.array(data)
#         kmeans = KMeans(n_clusters=k, init="random")
#         kmeans.fit(X)
#         cluster_labels = kmeans.labels_
#         clusters = [[] for _ in range(k)]

#         for device, label in zip(devices, cluster_labels):
#             clusters[label].append(device)
#         return clusters

#     def get_pe_data(self, pe):
        capacitance = sum(pe['capacitance'])
        handleSafeTask = pe['handleSafeTask']
        kind = sum(pe['acceptableTasks'])

        if pe['id'] != "cloud":
            devicePower = 0
            for index, core in enumerate(pe["voltages_frequencies"]):
                corePower = 0
                for mod in core:
                    freq, vol = mod
                    corePower += freq / vol
                devicePower += corePower
            devicePower = devicePower / pe['num_cores']
        else:
            devicePower = 1e9


        return [devicePower, capacitance,handleSafeTask,kind]