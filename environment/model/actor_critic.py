import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributions import Categorical
from sklearn.cluster import KMeans
import torch.optim as optim



class ActorCritic(nn.Module):
    def __init__(self,input_dim,output_dim,tree_max_depth,cirtic_input_dim,cirtic_hidden_layer_dim,devices,discount_factor=0):
        super(ActorCritic, self).__init__()
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.tree_max_depth=tree_max_depth 
        self.cirtic_input_dim=cirtic_input_dim 
        self.cirtic_hidden_layer_dim=cirtic_hidden_layer_dim 
        self.devices=devices 
        self.discount_factor=discount_factor
        
        self.actor = DDT(num_input=input_dim,num_output=output_dim,depth=0,max_depth=tree_max_depth)
        self.critic = nn.Sequential(
            nn.Linear(cirtic_input_dim,cirtic_hidden_layer_dim), 
            nn.ReLU(),
            nn.Linear(cirtic_hidden_layer_dim, 1)
        )
        self.rewards = []
        self.actions = []
        self.states = []
        self.pis = []

    def archive(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear_memory(self):
        self.rewards = []
        self.actions = []
        self.states = []
        self.pis = []

    def forward(self, x):
        p, path = self.actor(x)
        v = self.critic(x)
        return p, path, v

    def choose_action(self, observation):
        state = torch.tensor(observation, dtype=torch.float)
        pi, path,_ = self.forward(state)
        
        self.pis.append(pi)
        
        probs = F.softmax(pi, dim=-1)

        dist = Categorical(probs)
        action = dist.sample()

        return action.item(), path

    def calculate_returns(self):
        G = 0
        returns = []
        for reward in self.rewards[::-1]:
            G = G * self.discount_factor + reward
            returns.append(G)

        returns.reverse()
        returns = torch.tensor(returns, dtype=torch.float)
        return returns

    def calc_loss(self):
        states = torch.tensor(self.states, dtype=torch.float)
        actions = torch.tensor(self.actions, dtype=torch.float)
        returns = self.calculate_returns()
        values = []
        for state in states:
            v = self.critic(state)
            values.append(v)

        pis = torch.stack(self.pis, dim=0)
        values = torch.stack(values, dim=0).squeeze()

        probs = F.softmax(pis, dim=-1)

        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)

        # actor_loss = -log_probs * (returns - values)
        actor_loss = -torch.sum(log_probs * returns)
        critic_loss = F.mse_loss(values, returns)

        # total_loss = (actor_loss + critic_loss).mean()
        total_loss = actor_loss
        return total_loss


class CoreScheduler(nn.Module):
    def __init__(self,subtree_input_dims,subtree_max_depth, devices,subtree_lr=0.005):
        super(CoreScheduler, self).__init__()
        self.devices = devices
        self.forest = [
            self.createTree(input_dims=subtree_input_dims + device['num_cores'],
                            subtree_max_depth=subtree_max_depth,
                            output_dim=device['num_cores']*3)
            for device in devices]
        self.optimizers = [optim.Adam(tree.parameters(), lr=subtree_lr) for tree in self.forest]

    def createTree(self,input_dims ,subtree_max_depth,output_dim):
        return SubDDT(num_input=input_dims,num_output=output_dim,depth=0,max_depth=subtree_max_depth)


class DDT(nn.Module):
    def __init__(self, num_input, num_output, depth, max_depth):
        super(DDT, self).__init__()
        self.depth = depth
        self.max_depth = max_depth
        if depth != max_depth:
            self.weights = nn.Parameter(torch.empty(num_input).normal_(mean=0, std=0.1))
            self.bias = nn.Parameter(torch.zeros(1))
        if depth == max_depth:
            self.prob_dist = nn.Parameter(torch.zeros(num_output))
        if depth < max_depth:
            self.left = DDT(num_input, num_output, depth + 1,max_depth)
            self.right = DDT(num_input, num_output, depth + 1,max_depth)
        


    def forward(self, x, path=""):
        if self.depth == self.max_depth:
            return self.prob_dist, path
    
        val = torch.sigmoid(torch.matmul(x, self.weights) + self.bias)
        
        left_output, left_path = self.left(x, path + "L")
        right_output, right_path = self.right(x, path + "R")
        
        # Combine both left and right outputs
        output = val * right_output + (1 - val) * left_output
    
        # Combine both paths (though paths would still follow the stronger choice)
        final_path = right_path if val >= 0.5 else left_path
        
        return output, final_path


class SubDDT(nn.Module):
    def __init__(self, num_input, num_output, depth, max_depth):
        super(SubDDT, self).__init__()
        self.depth = depth
        self.max_depth = max_depth
        if depth != max_depth:
            self.weights = nn.Parameter(torch.empty(num_input).normal_(mean=0, std=0.1))
            self.bias = nn.Parameter(torch.zeros(1))
        if depth == max_depth:
            self.prob_dist = nn.Parameter(torch.zeros(num_output))
        if depth < max_depth:
            self.left = SubDDT(num_input, num_output, depth + 1,max_depth)
            self.right = SubDDT(num_input, num_output, depth + 1,max_depth)
        
    def forward(self, x):
        if self.depth == self.max_depth:
            return self.prob_dist
        val = torch.sigmoid(torch.matmul(x, self.weights) + self.bias)
        if val >= 0.5:
            return val *  self.right(x)
        else:
            return 1-val * self.left(x)

