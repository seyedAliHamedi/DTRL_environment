import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from data.db import Database
from environment.dtrl.core_scheduler import CoreScheduler
from environment.dtrl.device_scheduler import DeviceScheduler
from environment.state import State

class Agent:

    def __init__(self, devices):
        self.devices = devices
        self.core = CoreScheduler(devices)
        self.device = DeviceScheduler(devices)
        # self.value = ValueNetwork()
        self.gamma = 0.95
        self.db = Database()

        self.log_probs = {}
        self.device_values = {}
        self.core_values = {}
        self.rewards = {}
    
    def value_net(self):
        return 1
    
    
    def take_action(self,state,task_queue):
        job_state , pe_state = state
        current_task = task_queue[0]
        input_state = self.get_input(current_task,pe_state)
        input_state = torch.tensor(input_state, dtype=torch.float32)
    
        option_logits,devices = self.device.agent(input_state)
        option_probs = F.softmax(option_logits, dim=-1)
        option = torch.multinomial(option_probs, num_samples=1).squeeze().item()

        selected_device = devices[option]
        selected_device_index = self.devices.index(selected_device)

        sub_state = self.get_state(current_task,[selected_device])
        action_logits = self.core.agent(sub_state,selected_device_index)
        action_probs = F.softmax(action_logits, dim=-1)
        action = torch.multinomial(action_probs, num_samples=1).squeeze().item()

        log_prob = F.log_softmax(action_logits, dim=-1)[action]

        device_value,core_value = self.value_net(state)
        reward = self.takeAction()

        self.log_probs[current_task["job_ID"]].append(log_prob)
        self.device_values[current_task["job_ID"]].append(device_value)
        self.core_values[current_task["job_ID"]].append(core_value)
        self.rewards[current_task["job_ID"]].append(reward)


        for job in job_state:
            if job.is_finished() :
                self.update_params(job.ID,)
                del self.log_probs[current_task["job_ID"]]
                del self.device_values[current_task["job_ID"]]
                del self.core_values[current_task["job_ID"]]
                del self.rewards[current_task["job_ID"]]
    
    


     
    def updata_params(self, job_id,log_probs,core_values, rewards):
    
        advantages, returns = self.compute_advantages(rewards, core_values, gamma=self.gamma)
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
    
        print(
            f"Job: {job_id} Finished, Policy Loss: {policy_loss.item()}, Value Loss: {core_values_loss.item()}"
        )
    
    def compute_advantages(self,rewards, values, gamma=0.99, normalize=True):
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
    
    def get_input(self,task,pe_dict_list):
        task_features = self.get_task_data(task)
        pe_features = []
        for pe_dict in pe_dict_list:
            pe_features.extend(self.get_pe_data(pe_dict))
        return task_features+pe_features
    
    def get_task_data(self,task):
        return [
            task["compution_load"],
            task["input_size"],
            task["output_size"],
            task["task_kind"],
            task["is_safe"],
        ]
    
    def get_pe_data(self, pe_dict):
        pe = self.db.get_device(pe_dict["id"])
        battery_capacity = pe["battery_capacity"]
        battery_level = pe_dict["battery_level"]
        battery_isl = pe["isl"]
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

    