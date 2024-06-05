from environment.dtrl.value_network import ValueNetwork
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from data.db import Database
from environment.dtrl.core_scheduler import CoreScheduler
from environment.dtrl.device_scheduler import DeviceScheduler
from environment.state import State
from environment.utilities.window_manager import Preprocessing
import torch.optim as optim


class Agent:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)

            devices = Database.get_all_devices()
            cls._instance.devices = devices
            cls._instance.core = CoreScheduler(devices)
            cls._instance.device = DeviceScheduler(devices)
            net = ValueNetwork(input_size=4 * len(devices) + 5)
            cls._instance.value_net = net
            cls._instance.value_optimizer = optim.Adam(net.parameters(), lr=0.005)
            cls._instance.criterion = nn.MSELoss()

            cls._instance.gamma = 0.95

            cls._instance.log_probs = {}
            cls._instance.values = {}
            cls._instance.rewards = {}

        return cls._instance

    def run(self):
        queue = Preprocessing().get_agent_queue()

        print(f"Agent  queue: {queue}")
        for job_ID in queue.keys():
            task_queue = queue[job_ID]
            self.schedule(task_queue, job_ID)

    def schedule(self, task_queue, job_id):
        if len(task_queue) == 0:
            return

        if job_id not in self.log_probs:
            self.log_probs[job_id] = []
            self.values[job_id] = []
            self.rewards[job_id] = []

        job_state, pe_state = State().get()

        current_task_id = task_queue.pop(0)
        Preprocessing().remove_from_queue(current_task_id)

        current_task = Database().get_task(current_task_id)
        input_state = get_input(current_task, pe_state)
        input_state = torch.tensor(input_state, dtype=torch.float32)

        option_logits = self.device.agent(input_state)
        current_devices = self.device.agent.get_devices(input_state)
        option_dist = torch.distributions.Categorical(F.softmax(option_logits, dim=-1))
        option = option_dist.sample().item()

        selected_device = current_devices[option]
        selected_device_index = self.devices.index(selected_device)

        if selected_device['type'] != "cloud":
            sub_state = get_input(current_task, {0: pe_state[selected_device['id']]})
            action_logits = self.core.forest[selected_device_index](sub_state)
            action_dist = torch.distributions.Categorical(F.softmax(action_logits, dim=-1))
            action = action_dist.sample().item()
            selected_core_index = action
            selected_core = selected_device["voltages_frequencies"][selected_core_index]
            dvfs = selected_core[np.random.randint(0, 3)]

        if selected_device['type'] == "cloud":
            selected_core_index = -1
            i = np.random.randint(0, 1)
            dvfs = [(50000, 13.85), (80000, 24.28)][i]

        print(
            f"Agent Action::Device: {selected_device_index} | Core: {selected_core_index} | freq: {dvfs[0]} | vol: {dvfs[1]} | task_id: {current_task_id} | cl: {Database.get_task(current_task_id)['computational_load']} \n")
        reward = State().apply_action(selected_device_index, selected_core_index, dvfs[0], dvfs[1], current_task_id)

        return
        value = self.value_net(input_state)
        if len(task_queue) <= 0:
            temp_features = [0, 0, 0, 0, 0]
            _, next_pe_state = State().get()
            for pe in next_pe_state.values():
                temp_features.extend(get_pe_data(pe))
            next_input_state = torch.tensor(temp_features, dtype=torch.float32)
            next_value = self.value_net(next_input_state)
        else:
            next_current_task_id = task_queue[0]
            next_current_task = Database().get_task(current_task_id)
            _, next_pe_state = State().get()
            next_input_state = get_input(next_current_task, next_pe_state)
            next_input_state = torch.tensor(next_input_state, dtype=torch.float32)
            next_value = self.value_net(next_input_state)

        target = reward + self.gamma * next_value.item()
        target = torch.tensor([target])
        advantage = target - value

        option_loss = -option_dist.log_prob(torch.tensor(option)) * advantage
        actor_loss = -action_dist.log_prob(torch.tensor(action)) * advantage

        print(f"loss : {actor_loss + option_loss}")

        critic_loss = self.criterion(value, target)

        self.device.optimizer.zero_grad()
        option_loss.backward(retain_graph=True)
        self.device.optimizer.step()

        self.core.optimizers[selected_device_index].zero_grad()
        actor_loss.backward()
        self.core.optimizers[selected_device_index].step()

        # Update critic network
        self.value_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.value_optimizer.step()

        job_state = State().get_job(job_id)
        if len(job_state["remainingTasks"]) == 0:
            del self.log_probs[job_id]
            del self.device_values[job_id]
            del self.core_values[job_id]
            del self.rewards[job_id]

        # TODO : local scheduler

    def updata_params(self, job_id, log_probs, core_values, rewards):

        advantages, returns = self.compute_advantages(rewards, core_values, )
        policy_loss = -(torch.stack(log_probs) * advantages.detach()).sum()

        # device_value_loss = F.mse_loss(torch.stack(device_value).squeeze(), returns.detach())
        core_values_loss = F.mse_loss(torch.stack(core_values).squeeze(), returns.detach())

        self.device.optimizer.zero_grad()
        for optimizer in self.core.optimizers:
            optimizer.zero_grad()

        policy_loss.backward()

        self.device.optimizer.step()
        for optimizer in self.core.optimizers:
            optimizer.step()

        # value.zero_grad()
        # device_value_loss.backward()
        core_values_loss.backward()
        # optimizer_value.step()

        print(f"Job: {job_id} Finished, Policy Loss: {policy_loss.item()}, Value Loss: {core_values_loss.item()}")

    def compute_advantages(self, rewards, values, gamma=0.99, normalize=True):
        advantages = []
        returns = []
        advantage = 0

        for t in reversed(range(len(rewards))):
            td_error = (
                    rewards[t]
                    + gamma * (0 if t == len(rewards) - 1 else values[t + 1])
                    - values[t]
            )
            advantage = td_error + gamma * 0.95 * advantage
            advantages.insert(0, advantage)
            returns.insert(0, advantage + values[t])

        advantages = torch.tensor(advantages)
        returns = torch.tensor(returns)

        if normalize:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns


####### UTILITY #######     
def get_input(task, pe_dict):
    task_features = get_task_data(task)
    pe_features = []
    for pe in pe_dict.values():
        pe_features.extend(get_pe_data(pe))
    return task_features + pe_features


def get_task_data(task):
    return [
        task["computational_load"],
        task["input_size"],
        task["output_size"],
        task["task_kind"],
        task["is_safe"],
    ]


def get_pe_data(pe_dict):
    pe = Database.get_device(pe_dict["id"])
    battery_capacity = pe["battery_capacity"]
    battery_level = pe_dict["batteryLevel"]
    battery_isl = pe["ISL"]
    battery = (battery_level / battery_capacity - battery_isl) * battery_capacity

    num_cores = pe["num_cores"]
    cores_availability = pe_dict["occupiedCores"]
    cores = 1 - (sum(cores_availability) / num_cores)

    devicePower = 0
    for index, core in enumerate(pe["voltages_frequencies"]):
        if cores_availability[index] == 1:
            continue
        corePower = 0
        for mod in core:
            freq, vol = mod
            corePower += freq / vol
        devicePower += corePower
    devicePower = devicePower / num_cores

    error_rate = pe["error_rate"]

    return [cores, devicePower, battery, error_rate]
