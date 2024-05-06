import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from environment.dtrl.core_scheduler import CoreScheduler
from environment.dtrl.device_scheduler import DeviceScheduler

class Agent:

    def __init__(self, devices):
        self.devices = devices
        self.core = CoreScheduler(devices)
        self.device = DeviceScheduler(devices)
        # self.value = ValueNetwork()
        self.gamma = 0.95
    
    def value_net(self):
        return 1
    def takeAction(self):
        return 1
    
    def schedule(self,task_queue,env_snapshot,active_jobs):
        log_probs = {}
        device_values = {}
        core_values = {}
        rewards = {}
        for taskToken in range(task_queue):
            task  = taskToken.task
            state = self.get_state(task,env_snapshot)
            state = torch.tensor(state, dtype=torch.float32)
    
            option_logits,devices = self.device.agent(state)
            option_probs = F.softmax(option_logits, dim=-1)
            option = torch.multinomial(option_probs, num_samples=1).squeeze().item()
    
            selected_device = devices[option]
            selected_device_index = self.devices.index(selected_device)
    
            sub_state = self.get_state(task,[selected_device])
            action_logits = self.core.agent(sub_state,selected_device_index)
            action_probs = F.softmax(action_logits, dim=-1)
            action = torch.multinomial(action_probs, num_samples=1).squeeze().item()
    
            log_prob = F.log_softmax(action_logits, dim=-1)[action]
    
            device_value,core_value = self.value_net(state)
            reward, done = self.takeAction()
    
            log_probs[task.job_ID].append(log_prob)
            device_values[task.job_ID].append(device_value)
            core_values[task.job_ID].append(core_value)
            rewards[task.job_ID].append(reward)
    
            for job in active_jobs:
                if job.is_finished() :
                    self.update_params(
                        job.ID,
                        log_probs[task.job_ID],
                        device_values[task.job_ID],
                        core_values[task.job_ID],
                        rewards[task.job_ID],
                    )
                    del log_probs[task.job_ID]
                    del device_values[task.job_ID]
                    del core_values[task.job_ID]
                    del rewards[task.job_ID]
    
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
    
    def get_state(self,task,snapshot):
        task_features = self.get_task_data(task)
        pe_features = []
        for device in snapshot:
            pe_features.extend(self.get_pe_data(device))
        return task_features+pe_features
    
    def get_task_data(self,task):
        return[
            task.compution_load,
            task.input_size,
            task.output_size,
            task.task_kind,
            task.is_safe,
        ]
    
    def get_pe_data(self, pe):
        battery_capacity = pe.battery_capacity
        battery_level = pe.battery_eval
        battery_isl = pe.isl
        battery = (battery_level / battery_capacity - battery_isl) * battery_capacity
    
        num_cores = pe.num_cores
        cores_availability = self.cores_availability
        cores = 1 - (sum(cores_availability) / num_cores)
    
        devicePower = 0
        for index, core in enumerate(pe.cores_attrs["voltages_frequencies"]):
            if cores_availability[index] == 1:
                continue
            corePower = 0
            for mod in core:
                freq, vol = mod
                corePower += freq / vol
            devicePower += corePower
        devicePower = devicePower / num_cores
    
        error_rate = pe.error_rate
    
        return [cores, devicePower, battery, error_rate]

    @classmethod
    def core_schedule(cls, cores_attrs, cores_availability, queue):
        # ! change the queue
        # ! change the core avail
        # queue = [  token   ]

        # cores_attrs = {
        #             "num_of_cores" (int),
        #             "voltages_frequencies" (vector,(int)),
        #             "capacitance" (vector,(int)),
        #             "power_idle (vector,(int))"
        #         }

        # command = {
        # (core)0 : token, cpu_performance_mod
        #       1 : token, cpu_performance_mod
        #       2 : token, cpu_performance_mod
        #             .
        #             .
        #       n: token, cpu_performance_mod
        #   }
        command = {}
        log = []
        if len(queue) == 0:
            return command, log
        for item_index, item in enumerate(queue):
            free_core_index = -1
            for index, core in enumerate(cores_availability):
                if cores_availability[index] == 0:
                    free_core_index = index
                    break
            if free_core_index == -1:
                log.append(f"{item.task.name} -> waiting for free core")
                return command, log
            else:
                log.append(f"{item.task.name} -> core{free_core_index}")
                command[free_core_index] = [queue.pop(item_index), 1]
                cores_availability[free_core_index] = 1
        return command, log
