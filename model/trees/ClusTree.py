import torch
from collections import Counter
from sklearn.cluster import KMeans
import numpy as np
import torch.nn as nn
import torch.optim as optim

from configs import learning_config

class ClusTree(nn.Module):
    def __init__(self, num_input, devices, depth, max_depth, counter=0, exploration_rate=0):
        """
        Initializes the ClusTree structure.
        
        Args:
            num_input: Number of input features.
            devices: List of devices to be clustered.
            depth: Current depth of the tree.
            max_depth: Maximum depth of the tree.
            num_epoch: Number of epochs for training or simulation.
            counter: Counter for exploration steps.
            exploration_rate: Initial exploration rate for the tree.
        """
        super(ClusTree, self).__init__()
        
        # Tree properties
        self.depth = depth
        self.max_depth = max_depth
        self.devices = devices
        self.counter = counter
        
        # Exploration parameters
        self.epsilon = learning_config['explore_epsilon']
        num_epoch = learning_config['num_epoch']
        
        self.exp_mid_bound = num_epoch * self.epsilon
        self.exploration_rate = self.exp_mid_bound + self.exp_mid_bound / 2
        self.exp_threshold = self.exp_mid_bound - self.exp_mid_bound / 2
        self.shouldExplore = learning_config['should_explore']  # Use boolean for clarity
        
        # Leaf node: initialize the probability distribution for devices
        if depth == max_depth:
            self.logit_regressor = nn.Sequential(
                nn.Linear(learning_config['pe_num_features'],128),
                nn.Sigmoid(),
                nn.Linear(128,128),
                nn.Sigmoid(),
                nn.Linear(128,1),
            )
            self.logit_optimizer = optim.Adam(self.logit_regressor.parameters(),lr=0.01)
            self.prob_dist = nn.Parameter(torch.ones(len(devices)))  # Leaf stores device probabilities
        else:
            # Internal node: initialize weights and bias for splitting decision
            self.weights = nn.Parameter(torch.empty(num_input).normal_(mean=0, std=0.1))
            self.bias = nn.Parameter(torch.zeros(1))
            
            # Cluster the devices and create child nodes
            clusters = balance_kmeans_cluster(devices)
            self.left = ClusTree(num_input, clusters[0], depth + 1, max_depth, counter, exploration_rate)
            self.right = ClusTree(num_input, clusters[1], depth + 1, max_depth,  counter, exploration_rate)

      
        
    def forward(self, x, path=""):
        # Leaf node: return the probability distribution and devices
        if self.depth == self.max_depth:
            return self.prob_dist, path, self.devices
        
        # Internal node: compute decision value using weights and bias
        val = torch.sigmoid((torch.matmul(x, self.weights) + self.bias))
        
        # Exploration phase: adjust the value randomly
        if np.random.random() < self.exploration_rate and self.shouldExplore:
            val = self.explore(val)
        
        # Recursive decision: traverse left or right based on val
        if val >= 0.5:
            right_output, right_path, devices = self.right(x, path + "R")
            return val * right_output, right_path, devices
        else:
            left_output, left_path, devices = self.left(x, path + "L")
            return (1 - val) * left_output, left_path, devices

    def explore(self, val):
        self.counter += 1
        self.exploration_rate -= self.epsilon  # Reduce exploration rate over time
        
        # Stop exploration once the threshold is reached
        if self.exploration_rate < self.exp_threshold:
            self.shouldExplore = False
        
        # Return an adjusted value for exploration
        return 1 - val


# CLUSTERING
def balance_kmeans_cluster(devices, k=2, random_state=42):
    data = [extract_pe_data_for_clustering(device) for device in devices]
    if len(devices) < k:
        return [devices] * k
    X = np.array(data)
    kmeans = KMeans(n_clusters=k, init="random", random_state=random_state)
    kmeans.fit(X)

    cluster_labels = kmeans.labels_
    clusters = [[] for _ in range(k)]

    balanced_labels = balance_clusters(cluster_labels, k, len(devices))

    for device, label in zip(devices, balanced_labels):
        clusters[label].append(device)
    return clusters


def extract_pe_data_for_clustering(pe):
    is_safe = pe['is_safe']
    battery_capacity = pe['battery_capacity']
    kind = [1 in pe['acceptable_tasks'],2 in pe['acceptable_tasks'] , 3 in pe['acceptable_tasks'],4 in pe['acceptable_tasks']]

    devicePower = 0
    for index, core in enumerate(pe["voltages_frequencies"]):
        corePower = 0
        for mod in core:
            freq, vol = mod
            corePower += freq / vol
        devicePower += corePower
    devicePower = devicePower / pe['num_cores']

    return [devicePower,battery_capacity, is_safe]+kind


def balance_clusters(labels, k, n_samples):
    """
        Adjusts the initial cluster assignments to ensure clusters are balanced.
        """
    target_cluster_size = n_samples // k
    max_imbalance = n_samples % k  # Allowable imbalance due to indivisible n_samples

    cluster_sizes = Counter(labels)

    # List to store the indices of samples in each cluster
    cluster_indices = {i: [] for i in range(k)}

    # Populate the cluster_indices dictionary
    for idx, label in enumerate(labels):
        cluster_indices[label].append(idx)

    # Reassign samples to achieve balanced clusters
    for cluster in range(k):
        while len(cluster_indices[cluster]) > target_cluster_size:
            for target_cluster in range(k):
                if len(cluster_indices[target_cluster]) < target_cluster_size:
                    sample_to_move = cluster_indices[cluster].pop()
                    labels[sample_to_move] = target_cluster
                    cluster_indices[target_cluster].append(sample_to_move)
                    # Exit early if target sizes are met with allowable imbalance
                    if _clusters_balanced(cluster_indices, target_cluster_size, max_imbalance):
                        return labels
                    break

    return labels


def _clusters_balanced(cluster_indices, target_size, max_imbalance):
    imbalance_count = sum(abs(len(indices) - target_size) for indices in cluster_indices.values())
    return imbalance_count <= max_imbalance