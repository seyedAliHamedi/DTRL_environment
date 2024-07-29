import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


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

        probs = F.softmax(pi, dim=0)
        dist = Categorical(probs)
        action = dist.sample()

        return action.item()

    def calculate_returns(self):
        G = 0
        gamma = 0.99
        returns = []
        for reward in self.rewards[::-1]:
            G = G + gamma * reward
            returns.append(G)

        returns.reverse()
        returns = torch.tensor(returns, dtype=torch.float)
        return returns

    def calc_loss(self):
        states = torch.tensor(self.states, dtype=torch.float)
        actions = torch.tensor(self.actions, dtype=torch.float)
        returns = self.calculate_returns()

        pi, values = self.forward(states)
        values = values.squeeze()
        critic_loss = (returns-values)**2

        probs = F.softmax(pi, dim=0)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        actor_loss = -log_probs*(returns-values)

        total_loss = (critic_loss + actor_loss).mean()
        return total_loss


class DDT(nn.Module):
    def __init__(self, num_input, num_output, depth, max_depth, ):
        super(DDT, self).__init__()
        self.depth = depth
        self.max_depth = max_depth

        if depth != max_depth:
            self.weights = nn.Parameter(torch.empty(
                num_input).normal_(mean=0, std=0.1))
            self.bias = nn.Parameter(torch.zeros(1))
            self.alpha = nn.Parameter(torch.zeros(1))
        if depth == max_depth:
            self.prob_dist = nn.Parameter(torch.zeros(num_output))
        if depth < max_depth:
            self.left = DDT(num_input, num_output, depth + 1,
                            max_depth)
            self.right = DDT(num_input, num_output, depth + 1,
                             max_depth)

    def forward(self, x):
        if self.depth == self.max_depth:
            return self.prob_dist
        val = torch.sigmoid(
            self.alpha * (torch.matmul(x, self.weights) + self.bias))

        if val >= 0.5:
            right_output = self.right(x)
            return val * right_output
        else:
            left_output = self.left(x)
            return (1 - val) * left_output
