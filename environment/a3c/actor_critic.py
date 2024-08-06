import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from environment.a3c.DDT import DDT


class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions):
        super(ActorCritic, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.actor = DDT(num_input=input_dims,
                         num_output=n_actions, depth=0, max_depth=3)
        self.critic = nn.Sequential(
            nn.Linear(input_dims, 128), nn.ReLU(), nn.Linear(128, 1))

        self.rewards = []
        self.actions = []
        self.states = []

    def archive(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear_memory(self):
        self.rewards = []
        self.actions = []
        self.states = []

    def forward(self, x):
        p = self.actor(x)
        v = self.critic(x)
        return p, v

    def choose_action(self, ob):
        state = torch.tensor([ob], dtype=torch.float)
        pi, _ = self.forward(state)

        # Ensure numerical stability for softmax
        pi = pi - pi.max()
        probs = F.softmax(pi, dim=-1)

        dist = Categorical(probs)
        action = dist.sample()

        return action.item()

    def calculate_returns(self):
        G = 0
        gamma = 0
        returns = []
        for reward in self.rewards[::-1]:
            G = G * gamma + reward
            returns.append(G)

        returns.reverse()
        returns = torch.tensor(returns, dtype=torch.float)
        return returns

    def calc_loss(self):
        states = torch.tensor(self.states, dtype=torch.float)
        # Ensure actions are long type for indexing
        actions = torch.tensor(self.actions, dtype=torch.long)
        returns = self.calculate_returns()

        pis = []
        values = []
        for state in states:
            pi, value = self.forward(state)
            pis.append(pi)
            values.append(value)
        pis = torch.stack(pis, dim=0)
        values = torch.stack(values, dim=0).squeeze()

        # Ensure numerical stability for softmax
        pis = pis - pis.max(dim=-1, keepdim=True)[0]
        probs = F.softmax(pis, dim=-1)

        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        # actor_loss = -log_probs * (returns - values)
        actor_loss = -torch.sum(log_probs * returns)
        critic_loss = F.mse_loss(values, returns, reduction='none')

        # total_loss = (actor_loss + critic_loss).mean()
        total_loss = actor_loss
        return total_loss
