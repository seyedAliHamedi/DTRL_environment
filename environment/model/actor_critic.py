import random
from typing import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from sklearn.cluster import KMeans
import torch.optim as optim

from environment.util import Exploration


class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, devices):
        super(ActorCritic, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.exploration_rate=0.9
        self.explore_decay=0.9995
        self.actor = ClusTree(devices=devices, depth=0, max_depth=3,exploration_rate=self.exploration_rate,explore_decay=self.explore_decay)
        self.critic = nn.Sequential(
            nn.Linear (8 , 128), nn.ReLU(), nn.Linear(128, 1))

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
        p, path, devices = self.actor(x)
        v = self.critic(x)
        return p, path, devices, v

    def choose_action(self, ob):
        state = torch.tensor(ob, dtype=torch.float)
        pi, path, devices, _ = self.forward(state)

        # Ensure numerical stability for softmax
        pi = pi - pi.max()
        probs = F.softmax(pi, dim=-1)

        dist = Categorical(probs)
        action = dist.sample()

        return action.item(), path, devices

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
            pi, _, _, value = self.forward(state)
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
        self.num_features = 8
        self.forest = [self.createTree(device) for device in devices]
        self.optimizers = [optim.Adam(tree.parameters(), lr=0.005) for tree in self.forest]

    def createTree(self, device):
        max_depth=np.log2(device['num_cores'])
        num_input=self.num_features
        num_output=device['num_cores'] * 3
        return DDT(num_input,num_output,0,max_depth)


class DDT(nn.Module):
    def __init__(self, num_input, num_output, depth, max_depth, ):
        super(DDT, self).__init__()
        self.depth = depth
        self.max_depth = max_depth

        if depth != max_depth:
            self.weights = nn.Parameter(torch.empty(num_input).normal_(mean=0, std=0.1))
            self.bias = nn.Parameter(torch.zeros(1))
            self.alpha = nn.Parameter(torch.zeros(1))
        if depth == max_depth:
            self.prob_dist = nn.Parameter(torch.zeros(num_output))
        if depth < max_depth:
            self.left = DDT(num_input, num_output, depth + 1,max_depth)
            self.right = DDT(num_input, num_output, depth + 1,max_depth)

    def forward(self, x):
        if self.depth == self.max_depth:
            return self.prob_dist
        val = torch.sigmoid(
            self.alpha * (torch.matmul(x, self.weights) + self.bias))

        if val >= 0.5:
            return val *  self.right(x)
        else:
            return (1 - val) * self.left(x)


class ClusterTree(nn.Module):
    class ExplorationParams:
        def __init__(self, exploration_rate=0.9, explore_decay=0.995):
            self.exploration_rate = exploration_rate
            self.explore_decay = explore_decay
            
    def __init__(self, devices, depth, max_depth,exploration_params=None):
        super(ClusterTree, self).__init__()
        self.depth = depth
        self.max_depth = max_depth
        self.devices=devices
        self.exploration_params = exploration_params if exploration_params else DDT.ExplorationParams()
        
        # 5 weights for task and 2 for each device
        num_features = 5 + 2 * len(devices)

        if depth != max_depth:
            self.weights = nn.Parameter(torch.empty(
                num_features).normal_(mean=0, std=0.1))
            self.bias = nn.Parameter(torch.zeros(1))
        if depth == max_depth:
            self.prob_dist = torch.nn.Parameter(torch.ones(10))

        if depth < max_depth:
            clusters = self.cluster(self.devices)
            left_cluster = clusters[0]
            right_cluster = clusters[1]
            self.left = ClusterTree(left_cluster, depth + 1, max_depth)
            self.right = ClusterTree(right_cluster, depth + 1, max_depth)

    def forward(self, x, path=""):
        if self.depth == self.max_depth:
            return self.prob_dist, path, self.devices

        val = torch.sigmoid((torch.matmul(x, self.weights.t()) + self.bias))

        a = np.random.random()
        a = float("{:.6f}".format(a))
        if random.random() < self.exploration_params.exploration_rate:
            val = np.random.random()
            self.exploration_params.exploration_rate -= self.exploration_params.explore_decay 

        if val >= 0.5:
            indices = [self.devices.index(device)
                       for device in self.right.devices]
            temp = x[5:].view(-1, 2)
            indices_tensor = torch.tensor(indices)
            x = torch.cat((x[0:5], temp[indices_tensor].view(-1)), dim=0)
            right_output, right_path, devices = self.right(x, path + "R")
            return val * right_output, right_path, devices
        else:
            indices = [self.devices.index(device)
                       for device in self.left.devices]
            temp = x[5:].view(-1, 2)
            indices_tensor = torch.tensor(indices)
            x = torch.cat((x[0:5], temp[indices_tensor].view(-1)), dim=0)
            left_output, left_path, devices = self.left(x, path + "L")
            return val * left_output, left_path, devices

    def cluster(self, devices, k=2, random_state=42):
        torch.manual_seed(42)
        data = [self.extract_pe_data_for_clustering(device) for device in devices]
        if len(devices) < k:
            return [devices] * k
        X = np.array(data)
        kmeans = KMeans(n_clusters=k, init="random", random_state=random_state)
        kmeans.fit(X)

        cluster_labels = kmeans.labels_
        clusters = [[] for _ in range(k)]

        balanced_labels = self.balance_clusters(cluster_labels, k, len(devices))

        for device, label in zip(devices, balanced_labels):
            clusters[label].append(device)
        return clusters

    def balance_clusters(self, labels, k, n_samples):
        target_cluster_size = n_samples // k
        max_imbalance = n_samples % k  

        cluster_sizes = Counter(labels)
        cluster_indices = {i: [] for i in range(k)}

        for idx, label in enumerate(labels):
            cluster_indices[label].append(idx)

        for cluster in range(k):
            while len(cluster_indices[cluster]) > target_cluster_size:
                for target_cluster in range(k):
                    if len(cluster_indices[target_cluster]) < target_cluster_size:
                        sample_to_move = cluster_indices[cluster].pop()
                        labels[sample_to_move] = target_cluster
                        cluster_indices[target_cluster].append(sample_to_move)

                        # Exit early if target sizes are met with allowable imbalance
                        if self._clusters_balanced(cluster_indices, target_cluster_size, max_imbalance):
                            return labels
                        break

        return labels

    def _clusters_balanced(self, cluster_indices, target_size, max_imbalance):
        imbalance_count = sum(abs(len(indices) - target_size) for indices in cluster_indices.values())
        return imbalance_count <= max_imbalance

    def extract_pe_data_for_clustering(self, pe):
        battery_capacity = pe['battery_capacity']
        battery_isl = pe['ISL']
        battery = (1 - battery_isl) * battery_capacity

        num_cores = pe['num_cores']

        devicePower = 0
        for index, core in enumerate(pe["voltages_frequencies"]):
            corePower = 0
            for mod in core:
                freq, vol = mod
                corePower += freq / vol
            devicePower += corePower
        devicePower = devicePower / num_cores

        error_rate = pe['error_rate']

        # return [ num_cores,devicePower, battery]
        return [ devicePower, battery]

