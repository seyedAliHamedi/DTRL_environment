import copy
import random
from collections import deque

import numpy as np
import torch
from matplotlib import pyplot as plt


class Agent:

    def __init__(self, input_size, output_size, learning_rate=0.0008, target_network=False, epsilon=1, gamma=0.9,
                 policy='softmax',
                 mem_size=1000, mini_batch_size=32, sync=200):
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(input_size, 150),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(150, 150),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(150, output_size),
            torch.nn.Softmax(dim=1),
        )

        # Second nn for stabilizing
        self.target_nn = copy.deepcopy(self.nn)
        self.target_network = target_network
        self.stabilizer_sync = sync
        self.current_sync = 0

        # Hyper params
        self.epsilon = epsilon
        self.gamma = gamma
        self.mem_size = mem_size
        self.mini_batch_size = mini_batch_size
        self.policy = policy

        # Attributes
        self.memory_buffer = deque(maxlen=mem_size)
        # self.loss_fn = torch.nn.MSELoss()
        self.action_size = output_size
        self.optimizer = torch.optim.Adam(self.nn.parameters(), lr=learning_rate)

        # Log
        self.loss_history = []

        # Custom

    def act(self, state):
        pred = self.nn(state)
        return pred

    def loss_fn(self, pred, r):
        return -1 * torch.sum(torch.log(pred) * r)

    def update(self, pred, r):
        loss = self.loss_fn(pred, r)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss_history.append(loss.item())

    def experience_replay(self):
        if len(self.memory_buffer) >= self.mini_batch_size:
            mini_batch = random.sample(self.memory_buffer, self.mini_batch_size)
            s1_batch = torch.cat([s1 for (s1, action, s2, reward, done) in mini_batch])
            action_batch = torch.Tensor([action for (s1, action, s2, reward, done) in mini_batch]).int()
            s2_batch = torch.cat([s2 for (s1, action, s2, reward, done) in mini_batch])
            reward_batch = torch.Tensor([reward for (s1, action, s2, reward, done) in mini_batch])
            done_batch = torch.Tensor([done for (s1, action, s2, reward, done) in mini_batch]).int()
            Q1 = self.nn(s1_batch)
            with torch.no_grad():
                if self.target_network:
                    Q2 = self.target_nn(s2_batch)
                else:
                    Q2 = self.nn(s2_batch)
            Q2_max = torch.max(Q2, dim=1)[0]
            X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
            Y = (reward_batch + self.gamma * ((1 - done_batch) * Q2_max)).squeeze()
            self.update(X, Y)

    def sync_with_stabilizer(self):
        if self.target_network is False:
            return
        self.current_sync += 1
        if self.current_sync % self.stabilizer_sync == 0:
            self.target_nn.load_state_dict(self.nn.state_dict())

    def add_experience(self, experience):
        self.memory_buffer.append(experience)

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
