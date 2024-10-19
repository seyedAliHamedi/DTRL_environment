import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from configs import learning_config
from environment.util import extract_pe_data
from model.utils import *

import torch.multiprocessing as mp





# Main Actor-Critic class, combining both the Actor and Critic networks
class ActorCritic(nn.Module):
    def __init__(self, devices,old_log_probs_global):
        super(ActorCritic, self).__init__()
        self.discount_factor = learning_config['discount_factor']
        self.devices = devices
        self.actor = get_tree(self.devices)  # Initialize actor
        self.critic = get_critic()  # Critic could be None
        self.checkpoint_file = learning_config['checkpoint_file_path']
        self.reset_memory()
        self.old_log_probs_global = old_log_probs_global

    # Store experiences in memory
    def archive(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    # Clear memory after an episode
    def reset_memory(self):
        self.rewards, self.actions, self.states, self.pis = [], [], [], []

    # Forward pass through both actor and critic (if present)
    def forward(self, x):
        # Get policy distribution and path from the actor
        # Determine if the actor is ClusTree or DDT based on its output
        if isinstance(self.actor, ClusTree):
            p, path, devices = self.actor(x)  # Get policy distribution and path from the actor
        else:
            p, path = self.actor(x)  # Get policy distribution and path from the actor
            devices = None  # tree does not return devices

        v = self.critic(x) if self.critic is not None else None  # Value estimate from the critic, if applicable
        return p, path, devices, v

   # Select action based on actor's policy
    def choose_action(self, observation):
        state = torch.tensor(observation, dtype=torch.float)
        pi, path, devices, _ = self.forward(state)

        self.pis.append(pi)  # Store policy distribution

        probs = F.softmax(pi, dim=-1)

        dist = Categorical(probs)  # Create a categorical distribution over actions
        action = dist.sample()

        return action.item(), path, devices  # Return sampled action, the path, and devices

    # Calculate the discounted returns from the stored rewards
    def calculate_returns(self):
        G = 0
        returns = []
        for reward in reversed(self.rewards):
            G = G * self.discount_factor + reward  # Discounted return calculation
            returns.append(G)
        returns.reverse()
        return torch.tensor(returns, dtype=torch.float)

    # Compute the actor-critic loss
    def calc_loss(self):
        states = torch.tensor(self.states, dtype=torch.float)
        actions = torch.tensor(self.actions, dtype=torch.long)
        returns = self.calculate_returns()


        # Stack the padded tensors
        pis = torch.stack(self.pis, dim=0)
        probs = F.softmax(pis, dim=-1)
        dist = Categorical(probs)

        # Log probabilities of the actions taken
        new_log_probs = dist.log_prob(actions)

        # Calculate advantages
        if self.critic is not None:
            values = torch.stack([self.critic(state) for state in states], dim=0).squeeze()
            advantages = returns - values.detach()  # Using detach to avoid gradient flow
        else:
            advantages = returns

        if learning_config['learning_algorithm'] == "ppo":
            # PPO objective

            # get log probs for the first time (no difference between old and new here)
            for i, action in enumerate(actions):
                if self.old_log_probs_global[action.item()] is None:
                    self.old_log_probs_global[action.item()] = Categorical(probs[i]).log_prob(action).item()

            # collect the correspanding old log probs
            old_log_probs = []
            for action in actions:
                old_log_probs.append(self.old_log_probs_global[action.item()])
            old_log_probs = torch.tensor(old_log_probs)

            ratio = torch.exp(new_log_probs - old_log_probs)  # Importance ratio
            # update log probs
            for i, action in enumerate(actions):
                self.old_log_probs_global[action.item()] = Categorical(probs[i]).log_prob(action).item()

            p1 = ratio * advantages
            p2 = torch.clamp(ratio, 1 - learning_config['ppo_epsilon'], 1 + learning_config['ppo_epsilon']) * advantages

            actor_loss = -torch.min(p1, p2).sum()  # Minimize the clipped objective
        else:
            actor_loss = -torch.sum(new_log_probs * advantages)

        critic_loss = 0
        if self.critic:
            critic_loss = F.mse_loss(values, returns)

        return actor_loss + critic_loss


    def update_regressor(self):
        def update_leaf_nodes(node):
            if node.depth == node.max_depth:
                devices = node.devices if isinstance(self.actor, ClusTree) else self.devices
                dist = node.prob_dist
                pe_data = torch.tensor(
                    [extract_pe_data(device) for device in devices],
                    dtype=torch.float32
                )
                pred = node.logit_regressor(pe_data)
                loss = F.mse_loss(pred.squeeze(), dist)
                node.logit_optimizer.zero_grad()
                loss.backward()
                node.logit_optimizer.step()
                return

            if node.left:
                update_leaf_nodes(node.left)
            if node.right:
                update_leaf_nodes(node.right)

        # Start from the root actor
        actor = self.actor
        update_leaf_nodes(actor)
