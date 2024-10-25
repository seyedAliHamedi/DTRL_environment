import numpy as np

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import StepLR

from environment.util import Utility, gini_coefficient
from model.actor_critic import ActorCritic
from model.utils import CoreScheduler

from configs import learning_config,environment_config
class Agent(mp.Process):
    def __init__(self, name, global_actor_critic, global_optimizer, barrier, shared_state):
        super(Agent, self).__init__()
        
        self.name = name  # worker/agent name
        self.devices = shared_state.db_devices  # devices from database
        self.util = Utility(devices=self.devices)
        self.global_actor_critic = global_actor_critic  # global shared actor-critic model
        self.global_optimizer = global_optimizer  # shared Adam optimizer
        self.scheduler = StepLR(self.global_optimizer, step_size=1000, gamma=0.9)

        # the local actor-critic and the core scheduler
           # Use the already updated global model's state dict
        self.local_actor_critic = ActorCritic(devices=self.devices)

       

        self.assigned_job = None  # current job assigned to the agent
        self.runner_flag = mp.Value('b', True)  # flag to control agent's execution loop
        self.barrier = barrier  # barrier for synchronization across processes
        self.state = shared_state  # shared state between agents
        # self.state.core = CoreScheduler(subtree_input_dims=10, subtree_max_depth=3, devices=self.devices, subtree_lr=0.005)
        self.time_out_counter = environment_config['time_out_counter']  # counter for timeout detection
        self.t_counter = 0  # tracks agent's wait time


    def init_logs(self):
        """Initialize logs for the agent."""
        self.state.agent_log[self.assigned_job] = {}
        self.reward_log, self.time_log, self.energy_log = [], [], []
        self.kind_fails_log = self.safe_fails_log = self.queue_fails_log = self.battery_fails_log = self.fails_log = 0
        self.iot_usuage = self.mec_usuage = self.cc_usuage = 0
        self.path_history = []


    def run(self):
        """Main loop where the agent keeps running, processing jobs."""
        while self.runner_flag:
            self.barrier.wait()  # wait for all agents to be synchronized
            if self.assigned_job is None:
                self.assigned_job = self.state.assign_job_to_agent()
                if self.assigned_job is None:
                    continue
                # reset the status of the agent    
                self.t_counter = 0
                self.local_actor_critic.reset_memory()
                self.init_logs()
                # utilization = [sum(usage) for usage in self.state.device_usuages ]
                # gin = gini_coefficient(utilization)
                # used_devices_count = sum(1 for usage in self.state.device_usuages  if 1 in usage)
                # diversity = used_devices_count / len(self.devices)
                # utilization = torch.tensor(utilization, dtype=torch.float)
            # retrive the agent task_queue
            try:
                task_queue = self.state.preprocessor.get_agent_queue().get(self.assigned_job)
            except:
                continue
            if task_queue is None:
                continue
            
            self.t_counter += 1
            if self.t_counter >= self.time_out_counter and current_job and len(current_job["runningTasks"]) ==0 and len(current_job["remainingTasks"]) ==0 :
                self._timeout_on_job()
                continue
            
           
            
            for task in task_queue:
                # self.schedule(task,gin,diversity,utilization)
                self.schedule(task)
                
                
            # Check if the current job is complete
            try:
                current_job = self.state.jobs[self.assigned_job]
            except:
                continue
            if current_job and len(current_job["runningTasks"]) + len(current_job["finishedTasks"]) == current_job["task_count"]:
                print(f"DONE")
                # start_time = time.time()
                self.update()
                # print(self.name,"TOOK " ,time.time()-start_time,"UPDATE UPDATE PARAMETERS")
                self.assigned_job = None

    def stop(self):
        self.runner_flag = False
    def add_device(self, new_device):
        """Add device with synchronized state dict update."""
        self.util = Utility(devices=self.devices)
        
    def remove_device(self, device_index):
        """Remove device with synchronized state dict update."""
        self.util = Utility(devices=self.devices)
        
    def schedule(self, current_task_id,gin=None,diversity=None,utilization=None):
            # retrieve the necessary data
        pe_state = self.state.PEs
        current_task = self.state.db_tasks[current_task_id]
        input_state = self.util.get_input(current_task,gin=None,diversity=None)

        action, path, devices = self.local_actor_critic.choose_action(input_state)
        selected_device_index = action
        if devices:
            selected_device_index = self.env.devices.index(devices[selected_device_index])
            
        selected_device = self.devices[selected_device_index]
        
        # # second-level schedule for non cloud PEs , select a core and a Voltage Frequency Pair
        # sub_state = self.get_input(current_task, {0: pe_state[selected_device['id']]},subtree=True)
        # sub_state = torch.tensor(sub_state, dtype=torch.float)
        # action_logits = self.state.core.forest[selected_device_index](sub_state)
        # action_dist = torch.distributions.Categorical(F.softmax(action_logits, dim=-1))
        # action = action_dist.sample()
        # selected_core_dvfs_index = action.item()
        # selected_core_index, dvfs = self._select_core_dvfs(selected_device,selected_core_dvfs_index)

        selected_core_index = 0
        (freq, vol) = selected_device['voltages_frequencies'][selected_core_index][0]
        # applying action on the state and retrieving the result
        reward, fail_flag, energy, time = self.state.apply_action(
            selected_device_index, selected_core_index, freq, vol, current_task_id)

        # update the core schudler forest
        # self.update_core_scheduler(selected_device_index,action_dist,action,reward)
        # archive the result to the agent 
        self.local_actor_critic.archive(input_state, selected_device_index, reward)

        # saving agent logs 
        self.update_agent_logs(reward, time, energy, fail_flag, selected_device, path)

    def update_core_scheduler(self,selected_device_index,action_dist,action,reward):
        sub_tree_loss = (-action_dist.log_prob(action) * reward)
        self.state.core.optimizers[selected_device_index].zero_grad()
        sub_tree_loss.backward()
        self.state.core.optimizers[selected_device_index].step()
    
    def update_agent_logs(self, reward, time, energy, fail_flag, selected_device, path):
        """Update the logs for the agent based on task performance."""
        self.reward_log.append(reward)
        self.time_log.append(time)
        self.energy_log.append(energy)
        self.fails_log += sum(fail_flag)

        if selected_device['type'] == "iot":
            self.iot_usuage += 1
        elif selected_device['type'] == "mec":
            self.mec_usuage += 1
        elif selected_device['type'] == "cloud":
            self.cc_usuage += 1

        if fail_flag[0]: self.safe_fails_log += 1
        if fail_flag[1]: self.kind_fails_log += 1
        if fail_flag[2]: self.queue_fails_log += 1
        if fail_flag[3]: self.battery_fails_log += 1

        self.path_history.append(path)

    def save_agent_log(self, loss):
        """Save the logs of the agent after processing a job."""
        job_length = len(self.energy_log)
        result = {
            "loss": loss,
            "reward": sum(self.reward_log) / job_length,
            "time": sum(self.time_log) / job_length,
            "energy": sum(self.energy_log) /job_length,
            "safe_fails": self.safe_fails_log /job_length,
            "kind_fails": self.kind_fails_log /job_length,
            "queue_fails": self.queue_fails_log /job_length,
            "battery_fails": self.battery_fails_log /job_length,
            "fails": self.fails_log /job_length,
            "iot_usuage": self.iot_usuage /job_length,
            "mec_usuage": self.mec_usuage /job_length,
            "cc_usuage": self.cc_usuage /job_length,
        }
        self.state.save_agent_log(self.assigned_job, result, self.path_history)


    def update(self):
        """Update the global actor-critic based on the local model."""
        if learning_config['learning_algorithm']!='ppo':
            learning_config['ppo_epochs']=1
        for epoch in range(learning_config['ppo_epochs']):
            loss = self.local_actor_critic.calc_loss()

            self.global_optimizer.zero_grad()
            loss.backward(retain_graph=(epoch <= learning_config['ppo_epochs'] - 1))

            torch.nn.utils.clip_grad_norm_(self.global_actor_critic.parameters(), max_norm=1.0)

            for local_param, global_param in zip(self.local_actor_critic.parameters(), self.global_actor_critic.parameters()):
                global_param._grad = local_param.grad

            self.global_optimizer.step()
        # self.scheduler.step()
        self.save_agent_log(loss.item())  # save agent's performance
        self.local_actor_critic.load_state_dict(self.global_actor_critic.state_dict())

    ####### UTILITY FUNCTIONS #######

    def _timeout_on_job(self):
        """Handle timeout when a job is taking too long."""
        print("JOB Stuck ",self.assigned_job)
        self.state.remove_job(self.assigned_job)
        if len(self.local_actor_critic.states)>1:
            self.update()
        self.assigned_job = None
        
    def _select_core_dvfs(self, selected_device, selected_core_dvfs_index):
        selected_core_index = int(selected_core_dvfs_index / 3)
        selected_core = selected_device["voltages_frequencies"][selected_core_index]
        dvfs = selected_core[selected_core_dvfs_index % 3]
        return selected_core_index, dvfs

    
    def get_pe_data(self, pe_dict, pe_id,subtree):
        pe = self.state.database.get_device(pe_id)
        devicePower = pe['devicePower']

        batteryLevel = pe_dict['batteryLevel']
        battery_capacity = pe['battery_capacity']
        battery_isl = pe['ISL']
        battery = ((1 - battery_isl) * battery_capacity - batteryLevel) / battery_capacity

        num_cores = pe['num_cores']
        cores = 1 - (sum(pe_dict['occupiedCores']) / num_cores)
        if subtree:
            return pe_dict['occupiedCores'] + [ devicePower, battery]
        else:
            return [cores, devicePower, battery]
