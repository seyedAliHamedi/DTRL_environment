import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans 


class DeviceScheduler:
    def __init__(self, devices):
        self.devices = devices
        self.num_tnum_featuresask_features = 5
        self.max_tree_depth = 5
        self.agent = DeviceDDTNode(self.num_features, self.devices,0, self.max_tree_depth)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=0.005)


class DeviceDDTNode(nn.Module):
    def __init__(self, num_features,devices, depth,max_depth):
        super(DeviceDDTNode, self).__init__()
        self.depth = depth
        self.devices = devices
        self.num_features = num_features * (1 + len(devices))

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
            self.left = DeviceDDTNode(self.num_features,left_cluster,depth+1,max_depth)
            self.right = DeviceDDTNode(self.num_features,right_cluster,depth+1,max_depth)

    def forward(self, x):
        if self.depth == self.max_depth:
            return self.prob_dist,self.devices

        val = torch.sigmoid(self.alpha * (torch.matmul(x, self.weights.t()) + self.bias))
        a = np.random.uniform(0, 1)
        if a<0.10:
            val = 1-val
        if val >= 0.5:
            indices = [self.devices.index(device)+1 for device in self.right.devices]
            indices.insert(0,0)
            x=x.view(self.num_features,-1)
            indices_tensor = torch.tensor(indices)
            x = x[indices_tensor]
            x = x.view(1,-1)
            return val * self.right(x)
        else:
            indices = [self.devices.index(device)+1 for device in self.left.devices]
            indices.insert(0,0)
            x=x.view(self.num_features,-1)
            indices_tensor = torch.tensor(indices)
            x = x[indices_tensor]
            x = x.view(1,-1)
            return (1-val) * self.left(x)

    def cluster(self, devices, k=2):
        data = [self.get_pe_data(device) for device in devices]
        X = np.array(data)
        kmeans = KMeans(n_clusters=k, init="random")
        kmeans.fit(X)
        cluster_labels = kmeans.labels_
        clusters = [[] for _ in range(k)]

        for device, label in zip(devices, cluster_labels):
            clusters[label].append(device)
        return clusters


    def get_pe_data(self, pe):
        battery_capacity = pe.battery_capacity
        battery_level = pe.battery_eval
        battery_isl = pe.isl
        battery = (battery_level / battery_capacity - battery_isl) * battery_capacity

        num_cores = pe.num_cores
        cores_availability = self.cores_availability
        cores = 1 - (sum(cores_availability) / num_cores)

        devicePower = 0
        for index, core in enumerate(pe.cores_attrs["voltages_frequencies"]):
            if cores_availability[index] == 1:
                continue
            corePower = 0
            for mod in core:
                freq, vol = mod
                corePower += freq / vol
            devicePower += corePower
        devicePower = devicePower / num_cores

        error_rate = pe.error_rate

        return [cores, devicePower, battery, error_rate]
