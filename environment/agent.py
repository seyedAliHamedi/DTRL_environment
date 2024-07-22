from environment.dtrl.og_tree import OGTree
from environment.dtrl.value_network import ValueNetwork
import torch.optim as optim
from environment.window_manager import Preprocessing
from utilities.monitor import Monitor
from environment.state import State
from environment.dtrl.device_scheduler import DeviceScheduler
from environment.dtrl.core_scheduler import CoreScheduler
from data.db import Database
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Agent:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            devices = Database.get_all_devices()
            cls._instance.devices = devices
            cls._instance.core = CoreScheduler(devices)
            # cls._instance.device = DeviceScheduler(devices)
            d = OGTree(5, len(devices), 0, 3)
            cls._instance.device = d
            cls._instance.device_optimizer = optim.Adam(
                d.parameters(), lr=0.005)

            net = ValueNetwork(input_size=4 * len(devices) + 5)
            cls._instance.value_net = net
            cls._instance.value_optimizer = optim.Adam(
                net.parameters(), lr=0.005)
            cls._instance.loss_criterion = nn.MSELoss()

            cls._instance.main_log_probs = {}
            cls._instance.sub_log_probs = {}
            cls._instance.rewards = {}
            cls._instance.energy = {}
            cls._instance.time = {}
            cls._instance.fail = {}
            cls._instance.selected_devices = {}

            cls._instance.done_tasks = 0
            cls._instance.first_t_loss = None
            cls._instance.last_t_loss = None

        return cls._instance

    def run(self):
        queue = Preprocessing().get_agent_queue()

        # print(f"Agent  queue: {queue}")
        for job_ID in queue.keys():
            task_queue = queue[job_ID]
            self.schedule(task_queue, job_ID)

    def schedule(self, task_queue, job_id):
        if len(task_queue) == 0:
            return

        if job_id not in self.main_log_probs:
            self.main_log_probs[job_id] = []
            self.sub_log_probs[job_id] = []
            self.rewards[job_id] = []
            self.time[job_id] = []
            self.energy[job_id] = []
            self.fail[job_id] = []
            self.selected_devices[job_id] = []

        job_state, pe_state = State().get()

        current_task_id = task_queue.pop(0)
        Preprocessing().remove_from_queue(current_task_id)

        current_task = Database().get_task(current_task_id)
        # input_state = get_input(current_task, pe_state)
        input_state = get_input(current_task, {})
        input_state = torch.tensor(input_state, dtype=torch.float32)

        # option_logits = self.device.agent(input_state)
        option_logits = self.device(input_state)
        # current_devices = self.device.agent.get_devices(input_state)
        option_dist = torch.distributions.Categorical(
            F.softmax(option_logits, dim=-1))
        option = option_dist.sample()

        selected_device = self.devices[option.item()]
        selected_device_index = option.item()
        # test

        if selected_device['type'] != "cloud":
            sub_state = get_input(
                current_task, {0: pe_state[selected_device['id']]})
            sub_state = torch.tensor(
                sub_state, dtype=torch.float32)
            action_logits = self.core.forest[selected_device_index](sub_state)
            action_dist = torch.distributions.Categorical(
                F.softmax(action_logits, dim=-1))
            action = action_dist.sample()
            selected_core_index = action.item()
            selected_core = selected_device["voltages_frequencies"][selected_core_index]
            dvfs = selected_core[np.random.randint(0, 3)]
            self.sub_log_probs[job_id].append(action_dist.log_prob(action))

        if selected_device['type'] == "cloud":
            selected_core_index = -1
            i = np.random.randint(0, 1)
            dvfs = [(50000, 13.85), (80000, 24.28)][i]

        self.done_tasks += 1

        if (self.done_tasks / 100000 * 100) % 2 == 1:
            print(self.done_tasks / 100000 * 100)

        Monitor().add_log(
            f"Agent Action::Device: {selected_device_index} |"
            f" Core: {selected_core_index} | freq: {dvfs[0]} |"
            f" vol: {dvfs[1]} | task_id: {current_task_id} |"
            f" cl: {Database.get_task(current_task_id)['computational_load']}", start='\n', end='')

        reward, fail_flag, energy, time = State().apply_action(selected_device_index,
                                                               selected_core_index, dvfs[0], dvfs[1], current_task_id)

        # option_loss = -option_dist.log_prob(torch.tensor(option)) * reward
        # actor_loss = -action_dist.log_prob(torch.tensor(action)) * reward

        # self.device.optimizer.zero_grad()
        # option_loss.backward(retain_graph=True)
        # self.device.optimizer.step()

        # if action_dist is not None and action is not None:
        #     self.core.optimizers[selected_device_index].zero_grad()
        #     actor_loss.backward(retain_graph=True)
        #     self.core.optimizers[selected_device_index].step()

        self.main_log_probs[job_id].append(option_dist.log_prob(option))

        self.rewards[job_id].append(reward)
        self.time[job_id].append(time)
        self.energy[job_id].append(energy)
        self.fail[job_id].append(fail_flag)
        self.selected_devices[job_id].append(selected_device_index)
        print("999999999")
        job_state = State().get_job(job_id)
        if job_state and len(job_state["remainingTasks"]) == 0:
            total_loss = self.update(self.main_log_probs[job_id], self.sub_log_probs[job_id],
                                     self.rewards[job_id], self.selected_devices[job_id])

            Monitor().add_agent_log(
                {
                    'loss': total_loss,
                    'reward': sum(self.rewards[job_id]) / len(self.rewards[job_id]),
                    'time': sum(self.time[job_id]) / len(self.time[job_id]),
                    'energy': sum(self.energy[job_id]) / len(self.energy[job_id]),
                    'fail': sum(self.fail[job_id]) / len(self.fail[job_id]),
                }
            )

            del self.main_log_probs[job_id]
            del self.sub_log_probs[job_id]
            del self.rewards[job_id]
            del self.time[job_id]
            del self.energy[job_id]
            del self.fail[job_id]
            del self.selected_devices[job_id]

    def update(self, main_log_probs, sub_log_probs, rewards, indices):
        gamma = 0.99

        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32)
        # returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        returns = returns * torch.ones_like(returns, requires_grad=True)
        saved_log_probs_epoch = torch.stack(saved_log_probs_epoch)
        main_loss = -torch.sum(main_log_probs * returns)

        self.device_optimizer.zero_grad()
        main_loss.backward()
        self.device_optimizer.step()
        for i, _ in enumerate(rewards):
            self.core.optimizers[indices[i]].zero_grad()
            sub_loss = -sub_log_probs[i] * rewards[i]
            sub_loss.backward()
            self.core.optimizers[indices[i]].step()
        total_loss = -torch.sum(sub_log_probs * returns) + - \
            torch.sum(main_log_probs * returns)
        return total_loss

    def updata_params(self, job_id, log_probs, core_values, rewards):
        advantages, returns = self.compute_advantages(rewards, core_values)
        policy_loss = -(torch.stack(log_probs) * advantages.detach()).sum()
        core_values_loss = F.mse_loss(torch.stack(
            core_values).squeeze(), returns.detach())

        self.device.optimizer.zero_grad()
        for optimizer in self.core.optimizers:
            optimizer.zero_grad()

        policy_loss.backward(retain_graph=True)
        self.device.optimizer.step()
        for optimizer in self.core.optimizers:
            optimizer.step()

        core_values_loss.backward()
        print(f"Job: {job_id} Finished, Policy Loss: {policy_loss.item()}, Value Loss: {core_values_loss.item()}")

    def compute_advantages(self, rewards, values, gamma=0.99, normalize=True):
        advantages = []
        returns = []
        advantage = 0

        for t in reversed(range(len(rewards))):
            td_error = (rewards[t] + gamma *
                        (0 if t == len(rewards) - 1 else values[t + 1]) - values[t])
            advantage = td_error + gamma * 0.95 * advantage
            advantages.insert(0, advantage)
            returns.inser


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
    battery = (battery_level / battery_capacity -
               battery_isl) * battery_capacity

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
