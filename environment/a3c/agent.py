import torch
import torch.multiprocessing as mp

from environment.a3c.actor_critic import ActorCritic


class Agent(mp.Process):
    def __init__(self, name, global_actor_critic, optimizer):
        super(Agent, self).__init__()
        self.global_actor_critic = global_actor_critic
        self.local_actor_critic = ActorCritic(
            self.global_actor_critic.input_dims, self.global_actor_critic.n_actions)
        self.optimizer = optimizer
        self.name = name

    def run(self):
