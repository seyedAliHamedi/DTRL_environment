import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from configs import learning_config
from sklearn.linear_model import LinearRegression

from environment.util import extract_pe_data


class DDT(nn.Module):
    def __init__(self, num_input, num_output, depth, max_depth, counter=0, exploration_rate=0):
        """
        Initializes the DDT structure.
        
        Args:
            num_input: Number of input features.
            num_output: Number of output classes or predictions.
            depth: Current depth of the tree.
            max_depth: Maximum depth of the tree.
            counter: Counter for exploration steps.
            exploration_rate: Initial exploration rate for the tree.
        """
        super(DDT, self).__init__()

        # Tree properties
        self.depth = depth
        self.max_depth = max_depth
        self.counter = counter
        
        # Exploration parameters
        self.epsilon = learning_config['explore_epsilon']
        num_epoch = learning_config['num_epoch']

        self.exp_mid_bound = num_epoch * self.epsilon
        self.exploration_rate = self.exp_mid_bound + self.exp_mid_bound / 2
        self.exp_threshold = self.exp_mid_bound - self.exp_mid_bound / 2
        self.shouldExplore = learning_config['should_explore']  # Use boolean for clarity

        if depth == max_depth:
            self.prob_dist = nn.Parameter(torch.ones(num_output))
            self.logit_regressor = nn.Sequential(
                nn.Linear(learning_config['pe_num_features'], 256),
                nn.Sigmoid(),
                nn.Linear(256, 1),
            )
            self.logit_optimizer = optim.Adam(self.logit_regressor.parameters(), lr=0.01)
        if depth < max_depth:
            # Initialize weights, bias, and child nodes
            self.weights = nn.Parameter(torch.empty(num_input).normal_(mean=0, std=0.1))
            self.bias = nn.Parameter(torch.zeros(1))
            # Create left and right child nodes
            self.left = DDT(num_input, num_output, depth + 1, max_depth, self.counter, self.exploration_rate)
            self.right = DDT(num_input, num_output, depth + 1, max_depth, self.counter, self.exploration_rate)

    def forward(self, x, path=""):
        
        # Leaf node: return the probability distribution
        if self.depth == self.max_depth:
            return self.prob_dist, path

        # Internal node: compute decision value using weights and bias
        val = torch.sigmoid((torch.matmul(x, self.weights) + self.bias))

        # Exploration phase: adjust the value randomly
        if np.random.random() < self.exploration_rate and self.shouldExplore:
            val = self.explore(val)

        # Recursive decision: traverse left or right based on val
        if val >= 0.5:
            right_output, right_path = self.right(x, path + "R")
            return val * right_output, right_path
        else:
            left_output, left_path = self.left(x, path + "L")
            return (1 - val) * left_output, left_path

    def get_prob_dist(self, x):
        # Leaf node: return the probability distribution
        if self.depth == self.max_depth:
            return self.prob_dist

        # Internal node: compute decision value using weights and bias
        val = torch.sigmoid((torch.matmul(x, self.weights) + self.bias))

        # Exploration phase: adjust the value randomly
        if np.random.random() < self.exploration_rate and self.shouldExplore:
            val = self.explore(val)

        # Recursive decision: traverse left or right based on val
        if val >= 0.5:
            return self.right(x)
        else:
            return self.left(x)

    def explore(self, val):
        self.counter += 1
        self.exploration_rate -= self.epsilon  # Reduce exploration rate over time

        # Stop exploration once the threshold is reached
        if self.exploration_rate < self.exp_threshold:
            self.shouldExplore = False

        # Return an adjusted value for exploration
        return 1 - val

    def add_device(self, new_device):
        """Add a new device to the tree."""
        if self.depth == self.max_depth:
            new_device_features = extract_pe_data(new_device)
            device_tensor = torch.tensor(new_device_features, dtype=torch.float32)
            new_device_dist = self.logit_regressor(device_tensor)

            avg_logit = sum(self.prob_dist) / len(self.prob_dist)
            print(f"{len(avg_logit)} Avg Logit: {avg_logit:.4f}, Predicted Logit: {new_device_dist,}")

            self.prob_dist = nn.Parameter(torch.cat((self.prob_dist, new_device_dist)))
        else:
            self.left.add_device(new_device)
            self.right.add_device(new_device)

    def remove_device(self, device_index):
        """Remove a device from the tree."""
        if self.depth == self.max_depth:
            self.prob_dist = nn.Parameter(torch.cat((self.prob_dist[:device_index], 
                                                   self.prob_dist[device_index + 1:])))
        else:
            self.left.remove_device(device_index)
            self.right.remove_device(device_index)
            
    def train_logit_regressor(self, devices):
        """Train the logit regressor using current probabilities."""
        if self.depth == self.max_depth:
            # Get features and current probabilities for all devices
            device_features = torch.tensor([extract_pe_data(device) for device in devices], 
                                        dtype=torch.float32)
            current_probs = self.prob_dist.detach()
            
            # Train regressor
            self.logit_optimizer.zero_grad()
            predicted_logits = self.logit_regressor(device_features).squeeze()
            loss = F.mse_loss(predicted_logits, current_probs)
            loss.backward()
            self.logit_optimizer.step()
            
            return loss.item()
        
        # Recursively train child nodes
        left_loss = self.left.train_logit_regressor(devices) if hasattr(self, 'left') else 0
        right_loss = self.right.train_logit_regressor(devices) if hasattr(self, 'right') else 0
        return (left_loss + right_loss) / 2