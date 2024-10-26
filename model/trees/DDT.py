import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from configs import learning_config
from environment.util import extract_pe_data

class DDT(nn.Module):
    def __init__(self, num_input, num_output, depth, max_depth, counter=0, exploration_rate=0):
        super(DDT, self).__init__()

        # Tree properties
        self.depth = depth
        self.max_depth = max_depth
        self.counter = counter
        self.train_counter = 0
        
        # Exploration parameters
        self.epsilon = learning_config['explore_epsilon']
        num_epoch = learning_config['num_epoch']
        self.exp_mid_bound = num_epoch * self.epsilon
        self.exploration_rate = self.exp_mid_bound + self.exp_mid_bound / 2
        self.exp_threshold = self.exp_mid_bound - self.exp_mid_bound / 2
        self.shouldExplore = learning_config['should_explore']  

        if depth == max_depth:
            # Initialize probability distribution with proper normalization
            self.prob_dist = nn.Parameter(torch.ones(num_output) / num_output)
            
            # Improved logit regressor architecture
            self.logit_regressor = nn.Sequential(
                nn.Linear(learning_config['pe_num_features'], 32),
                nn.ReLU(),
                nn.Linear(32, 1),
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
        """Add a new device with normalized probabilities."""
        if self.depth == self.max_depth:
            # Get features and predict initial probability
            new_device_features = extract_pe_data(new_device)
            device_tensor = torch.tensor(new_device_features, dtype=torch.float32)
            new_prob = self.logit_regressor(device_tensor)

            avg_logit = sum(self.prob_dist) / len(self.prob_dist)
            new_device_dist = 0.8*new_prob + 0.2*avg_logit

            self.prob_dist = nn.Parameter(torch.cat((self.prob_dist, new_device_dist)))
        else:
            self.left.add_device(new_device)
            self.right.add_device(new_device)

    def remove_device(self, device_index):
        """Remove a device with probability redistribution."""
        if self.depth == self.max_depth:
            with torch.no_grad():
                # Get probability of removed device
                removed_prob = self.prob_dist[device_index]
                
                # Create new distribution without the removed device
                remaining_probs = torch.cat((
                    self.prob_dist[:device_index],
                    self.prob_dist[device_index + 1:]
                ))
                
                # Calculate redistribution weights based on current probabilities
                weights = F.softmax(remaining_probs, dim=0)
                
                # Redistribute the removed probability
                remaining_probs += removed_prob * weights
                
                # Update probability distribution
                self.prob_dist = nn.Parameter(remaining_probs)
        else:
            self.left.remove_device(device_index)
            self.right.remove_device(device_index)
            
    def train_logit_regressor(self, devices):
        """Train logit regressor with improved stability."""
        if self.train_counter < 10:
            self.train_counter += 1
            return
        
        if self.depth == self.max_depth:
            device_features = torch.tensor([extract_pe_data(device) for device in devices], 
                                        dtype=torch.float32)
            
            # Normalize current probabilities
            current_probs = F.softmax(self.prob_dist, dim=0).detach()
            
            # Multiple training iterations with gradient accumulation
            for _ in range(3):
                self.logit_optimizer.zero_grad()
                predicted_logits = self.logit_regressor(device_features).squeeze()
                
                # Use KL divergence loss for better probability distribution learning
                predicted_probs = F.softmax(predicted_logits, dim=0)
                loss = F.kl_div(
                    predicted_probs.log(),
                    current_probs,
                    reduction='batchmean'
                )
                
                loss.backward()
                self.logit_optimizer.step()
            
            self.train_counter = 0
        else:
            self.left.train_logit_regressor(devices)
            self.right.train_logit_regressor(devices)