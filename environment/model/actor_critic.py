import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim

from environment.model.utilities import DDT

# Main Actor-Critic class, combining both the Actor and Critic networks
class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim, tree_max_depth, critic_input_dim, critic_hidden_layer_dim, discount_factor=0):
        super(ActorCritic, self).__init__()
        self.input_dim =input_dim
        self.output_dim =output_dim
        self.tree_max_depth =tree_max_depth
        self.critic_input_dim =critic_input_dim
        self.critic_hidden_layer_dim =critic_hidden_layer_dim
          
        self.discount_factor = discount_factor
        self.actor = DDT(input_dim, output_dim, 0, tree_max_depth)  # Decision tree-like actor
        self.critic = nn.Sequential(
            nn.Linear(critic_input_dim, critic_hidden_layer_dim),
            nn.ReLU(),
            nn.Linear(critic_hidden_layer_dim, 1)  # Output single value for value estimation
        )

        # Initialize memory for storing the experiences
        self.reset_memory()

    # Store experiences in memory
    def archive(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    # Clear memory after an episode
    def reset_memory(self):
        self.rewards, self.actions, self.states, self.pis = [], [], [], []

    # Forward pass through both actor and critic
    def forward(self, x):
        p, path = self.actor(x)  # Policy distribution from the actor
        v = self.critic(x)  # Value estimate from the critic
        return p, path, v

    # Select action based on actor's policy
    def choose_action(self, observation):
        state = torch.tensor(observation, dtype=torch.float)
        pi, path, _ = self.forward(state)
        
        self.pis.append(pi)  # Store policy distribution

        probs = F.softmax(pi, dim=-1)
        dist = Categorical(probs)  # Create a categorical distribution over actions
        action = dist.sample()

        return action.item(), path  # Return sampled action and the path

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
        actions = torch.tensor(self.actions, dtype=torch.float)
        returns = self.calculate_returns()

        # Get value estimates for all states
        values = torch.stack([self.critic(state) for state in states], dim=0).squeeze()

        # Stack stored policy distributions
        pis = torch.stack(self.pis, dim=0)
        probs = F.softmax(pis, dim=-1)
        dist = Categorical(probs)

        # Log probabilities of the actions taken
        log_probs = dist.log_prob(actions)

        # Actor loss (policy gradient) and critic loss (value estimation)
        actor_loss = -torch.sum(log_probs * returns)
        critic_loss = F.mse_loss(values, returns)

        # Total loss = actor loss + critic loss
        return actor_loss 
