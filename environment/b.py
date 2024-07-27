import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import gym
import numpy as np

# Define the Actor-Critic Network


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.actor(x), self.critic(x)

    def act(self, state):
        logits, _ = self.forward(state)
        prob = torch.softmax(logits, dim=-1)
        action = torch.multinomial(prob, num_samples=1).item()
        return action

    def evaluate(self, state, action):
        logits, value = self.forward(state)
        prob = torch.softmax(logits, dim=-1)
        log_prob = torch.log(prob.gather(1, action))
        return log_prob, value

# Worker process


def worker(global_model, optimizer, env_name, gamma, worker_id, update_global_iter):
    local_model = ActorCritic(
        global_model.fc1.in_features, global_model.actor.out_features)
    local_model.load_state_dict(global_model.state_dict())
    env = gym.make(env_name)
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32)

    while True:
        log_probs = []
        values = []
        rewards = []

        for _ in range(update_global_iter):
            action = local_model.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32)

            log_prob, value = local_model.evaluate(state.unsqueeze(
                0), torch.tensor([[action]], dtype=torch.int64))
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)

            state = next_state
            if done:
                state = env.reset()
                state = torch.tensor(state, dtype=torch.float32)
                break

        R = 0
        if not done:
            _, value = local_model.evaluate(state.unsqueeze(
                0), torch.tensor([[action]], dtype=torch.int64))
            R = value.item()

        actor_loss = 0
        critic_loss = 0
        returns = []

        for reward in rewards[::-1]:
            R = reward + gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32)
        values = torch.cat(values)

        advantage = returns - values
        critic_loss = advantage.pow(2).mean()
        actor_loss = -(torch.cat(log_probs) * advantage.detach()).mean()

        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()

        for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
            global_param.grad = local_param.grad
        optimizer.step()

        local_model.load_state_dict(global_model.state_dict())

# Main function


def main():
    env_name = 'CartPole-v1'
    state_dim = gym.make(env_name).observation_space.shape[0]
    action_dim = gym.make(env_name).action_space.n
    global_model = ActorCritic(state_dim, action_dim)
    global_model.share_memory()
    optimizer = optim.Adam(global_model.parameters(), lr=0.001)
    num_workers = 4
    update_global_iter = 5
    gamma = 0.99

    workers = []
    for worker_id in range(num_workers):
        worker_process = mp.Process(target=worker, args=(
            global_model, optimizer, env_name, gamma, worker_id, update_global_iter))
        worker_process.start()
        workers.append(worker_process)

    for worker_process in workers:
        worker_process.join()


if __name__ == '__main__':
    main()
