import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ValueNetwork(nn.Module):
    def __init__(self,num_features,devices):
        super(ValueNetwork, self).__init__()
        self.devices = devices
        self.device_net = nn.Sequential(
            nn.Linear(self.input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, len(devices)),
            nn.Softmax(dim=1),
        )
        self.device_optimizer = optim.Adam(self.device_net.parameters(), lr=0.005)
        self.core_networks = [
            nn.Sequential(
                nn.Linear(num_features*2, 128),
                nn.ReLU(),
                nn.Linear(128, device.num_cores),
            )
            for device in self.devices
        ]
        self.core_optimizers = [optim.Adam(net.parameters(), lr=0.005) for net in self.core_networks]

    def forward(self,x):
        device_por_dist = self.device_net(x)
        selected_device_index = torch.multinomial(device_por_dist, 1).squeeze().item()
        selected_device = self.devices[selected_device_index]

        task_data = x[:5]
        device_data = torch.tensor(self.get_pe_data(selected_device))
        core_prob_dist = self.core_networks[selected_device_index](torch.cat(task_data,device_data))
        selected_core_index = torch.multinomial(core_prob_dist, 1).squeeze().item()
        
        return selected_device_index,selected_core_index
