import copy
import random
from collections import deque

import numpy as np
import torch
from matplotlib import pyplot as plt


class Agent:

    def __init__(self, input_size, output_size, learning_rate=0.001, target_network=False, epsilon=1, gamma=0.9,
                 mem_size=1000, mini_batch_size=50, sync=500):
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(input_size, 150),
            torch.nn.ReLU(),
            torch.nn.Linear(150, 150),
            torch.nn.ReLU(),
            torch.nn.Linear(150, 150),
            torch.nn.ReLU(),
            torch.nn.Linear(150, output_size),
        )

        # Second nn for stabilizing
        self.nn_stabilizer = copy.deepcopy(self.nn)
        self.target_network = target_network
        self.sync = sync
        self.current_sync = 0

        # Hyper params
        self.epsilon = epsilon
        self.gamma = gamma

        # Attributes
        self.mem_size = mem_size
        self.batch_size = mini_batch_size
        self.mem = deque(maxlen=mem_size)
        self.loss_fn = torch.nn.MSELoss()
        self.action_size = output_size
        self.optimizer = torch.optim.Adam(self.nn.parameters(), lr=learning_rate)

        # Log
        self.loss_history = []

    def act(self, state):
        pred = self.nn(state)
        action = np.random.choice(self.action_size, p=pred.data.numpy().squeeze())
        return action, pred

    def experience_replay(self):
        if len(self.mem) >= self.batch_size:
            mini_batch = random.sample(self.mem, self.batch_size)
        else:
            mini_batch = random.sample(self.mem, len(self.mem))
        s1_batch = torch.cat([s1 for (s1, action, reward) in mini_batch])
        action_batch = torch.Tensor([action for (s1, action, reward) in mini_batch]).int()
        reward_batch = torch.Tensor([reward for (s1, action, reward) in mini_batch])
        Q1 = self.nn(s1_batch)
        X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze(dim=1)
        Y = reward_batch
        return self.update(X, Y)

    def add_experience(self, experience):
        self.mem.append(experience)

    def update(self, X, Y):
        loss = self.loss_fn(X, Y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss_history.append(loss.item())
        return loss.item()

    def save(self, path):
        torch.save({
            'model_state_dict': self.nn.state_dict(),
            'epsilon': self.epsilon,
            'gamma': self.gamma,
            'loss_history': self.loss_history
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.nn.load_state_dict(checkpoint['model_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.gamma = checkpoint['gamma']
        self.loss_history = checkpoint['loss_history']

    def plot_loss(self):
        plt.bar(range(len(self.loss_history)), self.loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss History')
        plt.grid(True)
        plt.show()
