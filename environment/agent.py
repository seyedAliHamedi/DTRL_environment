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

class Agent:

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)

            devices = Database.get_all_devices()
            cls._instance.devices = devices
            cls._instance.core = CoreScheduler(devices)
            cls._instance.device = DeviceScheduler(devices)
            # cls._instance.value_net = ValueNetwork()
            
            cls._instance.gamma = 0.95
    
            cls._instance.log_probs = {}
            cls._instance.device_values = {}
            cls._instance.core_values = {}
            cls._instance.rewards = {}

        return cls._instance

 
    
    
    def run(self,task_queue):
        if len(task_queue)==0:
            return 
        job_state , pe_state = State().get()

        current_task_id = task_queue[0]
        current_task = Database().get_task(current_task_id)
        input_state = get_input(current_task,pe_state)
        input_state = torch.tensor(input_state, dtype=torch.float32)
    
        option_logits = self.device.agent(input_state)
        current_devices = self.device.agent.get_devices(input_state)
        option_probs = F.softmax(option_logits, dim=-1)
        option = torch.multinomial(option_probs, num_samples=1).squeeze().item()
        
        selected_device = current_devices[option]
        selected_device_index = self.devices.index(selected_device)

        print({0:selected_device})
        sub_state = get_input(current_task,{0:pe_state[selected_device['id']]})
        action_logits = self.core(sub_state,selected_device_index)
        action_probs = F.softmax(action_logits, dim=-1)
        selected_core_index= action = torch.multinomial(action_probs, num_samples=1).squeeze().item()
        selected_core = selected_device["voltages_frequencies"][selected_core_index]
        
        
        #TODO : local scheduler
        dvfs = selected_core[0]

        print("???? ",selected_device_index,selected_core_index,dvfs[0],dvfs[1])
        return


        device_value,core_value = self.value_net()
        reward = State().apply_action(selected_device_index,selected_core_index,dvfs[0],dvfs[1],current_task['id'])

        self.log_probs[current_task["job_ID"]].append(action_probs)
        self.device_values[current_task["job_ID"]].append(device_value)
        self.core_values[current_task["job_ID"]].append(core_value)
        self.rewards[current_task["job_ID"]].append(reward)


        for job in job_state:
            if len(job["remainingTasks"]) == 0 :
                self.update_params(job["id"])
                del self.log_probs[job["id"]]
                del self.device_values[job["id"]]
                del self.core_values[job["id"]]
                del self.rewards[job["id"]]
    
    


     
    def updata_params(self, job_id,log_probs,core_values, rewards):
    
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
    




####### UTILITY #######     
def get_input(task,pe_dict):
    task_features = get_task_data(task)
    pe_features = []
    for pe in pe_dict.values():
        pe_features.extend(get_pe_data(pe))
    return task_features+pe_features

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
    