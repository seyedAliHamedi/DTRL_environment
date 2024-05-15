import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans 


class DeviceScheduler:
    def __init__(self, devices):
        self.devices = devices
        self.num_features = 5
        self.max_tree_depth = 5
        self.agent = DeviceDDTNode(self.devices,0, self.max_tree_depth)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=0.005)
        self.selected_device_group = None


class DeviceDDTNode(nn.Module):
    def __init__(self, devices, depth,max_depth):
        super(DeviceDDTNode, self).__init__()
        self.depth = depth
        self.devices = devices
        self.num_features = 5 + 4 * len(devices)
        self.max_depth = max_depth

        if depth != max_depth:
            self.weights = nn.Parameter(torch.zeros(self.num_features))
            self.bias = nn.Parameter(torch.zeros(1))
            self.alpha = nn.Parameter(torch.zeros(1))
        if depth == max_depth:
            self.prob_dist = nn.Parameter(torch.zeros(len(devices)))

        if depth<max_depth:
            clusters = self.cluster(self.devices)
            left_cluster = clusters[0]
            right_cluster = clusters[1]
            self.left = DeviceDDTNode(left_cluster,depth+1,max_depth)
            self.right = DeviceDDTNode(right_cluster,depth+1,max_depth)

    def forward(self, x):
        if self.depth == self.max_depth:
            self.selected_device_group=self.devices
            return self.prob_dist

        val = torch.sigmoid(self.alpha * (torch.matmul(x, self.weights.t()) + self.bias))
            
        if val >= 0.5:
            indices = [self.devices.index(device) for device in self.right.devices]
            temp=x[5:].view(-1,4)
            indices_tensor = torch.tensor(indices)
            x = torch.cat((x[0:5], temp[indices_tensor].view(-1)), dim=0)
            return val * self.right(x)
        else:
            indices = [self.devices.index(device) for device in self.left.devices]
            temp=x[5:].view(-1,4)
            indices_tensor = torch.tensor(indices)
            x = torch.cat((x[0:5], temp[indices_tensor].view(-1)), dim=0)
            return val * self.left(x)
        
    def get_devices(self,x):
        if self.depth == self.max_depth:
            return self.devices
        
        val = torch.sigmoid(self.alpha * (torch.matmul(x, self.weights.t()) + self.bias))

        if val >= 0.5:
            indices = [self.devices.index(device) for device in self.right.devices]
            temp=x[5:].view(-1,4)
            indices_tensor = torch.tensor(indices)
            x = torch.cat((x[0:5], temp[indices_tensor].view(-1)), dim=0)
            return self.right.get_devices(x)
        else:
            indices = [self.devices.index(device) for device in self.left.devices]
            temp=x[5:].view(-1,4)
            indices_tensor = torch.tensor(indices)
            x = torch.cat((x[0:5], temp[indices_tensor].view(-1)), dim=0)
            return self.left.get_devices(x)
        
    def cluster(self, devices, k=2):
        data = [self.get_pe_data(device) for device in devices]
        if len(devices)<k:
            return [devices]*k
        X = np.array(data)
        kmeans = KMeans(n_clusters=k, init="random")
        kmeans.fit(X)
        cluster_labels = kmeans.labels_
        clusters = [[] for _ in range(k)]

        for device, label in zip(devices, cluster_labels):
            clusters[label].append(device)
        return clusters


    def get_pe_data(self, pe):
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

        return [num_cores, devicePower, battery, error_rate]
