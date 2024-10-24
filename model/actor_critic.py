import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from configs import learning_config
from environment.util import extract_pe_data
from model.utils import *

# Main Actor-Critic class, combining both the Actor and Critic networks
class ActorCritic(nn.Module):
    def __init__(self, devices):
        super(ActorCritic, self).__init__()
        self.discount_factor = learning_config['discount_factor']
        self.devices = devices
        self.actor = get_tree(self.devices)  # Initialize actor
        self.critic = get_critic()  # Critic could be None
        self.checkpoint_file = learning_config['checkpoint_file_path']
        self.reset_memory()
        self.old_log_probs = [None for _ in range(len(self.devices))]
        
        self.clip_param = learning_config['ppo_epsilon']
        self.gae_lambda = learning_config['gae_lambda']

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

    def calc_loss(self):
        states = torch.tensor(self.states, dtype=torch.float)
        actions = torch.tensor(self.actions, dtype=torch.long)
        rewards = torch.tensor(self.rewards, dtype=torch.float)

        if self.critic is not None:
            values = self.critic(states).squeeze()
            next_value = self.critic(states[-1]).item()
            returns = self.compute_gae(rewards, values.detach(), next_value)
            advantages = returns - values.detach()
        else:
            returns = self.calculate_returns()
            advantages = returns

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        pis = torch.stack(self.pis, dim=0)
        probs = F.softmax(pis, dim=-1)
        dist = Categorical(probs)

        new_log_probs = dist.log_prob(actions)

        if learning_config['learning_algorithm'] == "ppo":
            old_log_probs = []
            for i, action in enumerate(actions):
                if self.old_log_probs[action.item()] is None:
                    self.old_log_probs[action.item()] = Categorical(probs[i]).log_prob(action).item()
                old_log_probs.append(self.old_log_probs[action.item()])
            old_log_probs = torch.tensor(old_log_probs)

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Update old log probs
            for i, action in enumerate(actions):
                self.old_log_probs[action.item()] = new_log_probs[i].item()

        
        else:
            actor_loss = -(new_log_probs * advantages).mean()

        critic_loss = 0
        if self.critic:
            critic_loss = F.mse_loss(values, returns)

        entropy = dist.entropy().mean()
        entropy_coef = 0.00  # Small entropy coefficient
        loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropy

        return loss
    def compute_gae(self, rewards, values, next_value):
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.discount_factor * next_value - values[step]
            gae = delta + self.discount_factor * self.gae_lambda * gae
            
            # gae = torch.clamp(gae, -10.0, 10.0)
            returns.insert(0, gae + values[step])
            next_value = values[step]
        return torch.tensor(returns)

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
