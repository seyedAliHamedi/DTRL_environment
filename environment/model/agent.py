import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from environment.model.actor_critic import ActorCritic
from environment.model.utilities import CoreScheduler


class Agent(mp.Process):
    def __init__(self, name, global_actor_critic, global_optimizer, barrier, shared_state, time_out_counter):
        super(Agent, self).__init__()
        
        self.name = name  # worker/agent name
        self.devices = shared_state.database.get_all_devices()  # devices from database
        self.global_actor_critic = global_actor_critic  # global shared actor-critic model
        self.global_optimizer = global_optimizer  # shared Adam optimizer
        
        # the local actor-critic and the core scheduler
        self.local_actor_critic = ActorCritic(
            global_actor_critic.input_dim,
            global_actor_critic.output_dim,
            global_actor_critic.tree_max_depth,
            global_actor_critic.critic_input_dim,
            global_actor_critic.critic_hidden_layer_dim,
            global_actor_critic.discount_factor
        ) # local actor-critic model
       

        self.assigned_job = None  # current job assigned to the agent
        self.runner_flag = mp.Value('b', True)  # flag to control agent's execution loop
        self.barrier = barrier  # barrier for synchronization across processes
        self.state = shared_state  # shared state between agents
        self.time_out_counter = time_out_counter  # counter for timeout detection
        self.t_counter = 0  # tracks agent's wait time
        self.state.core = CoreScheduler(subtree_input_dims=10, subtree_max_depth=3, devices=self.devices, subtree_lr=0.005)


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

            # retrive the agent task_queue
            task_queue = self.state.preprocessor.get_agent_queue().get(self.assigned_job)
            if task_queue is None:
                continue
            
            self.t_counter += 1
            if self.t_counter >= self.time_out_counter:
                self._timeout_on_job()
                continue
            for task in task_queue:
                self.schedule(task)
                
            # Check if the current job is complete
            current_job = self.state.get_job(self.assigned_job)
            if current_job and len(current_job["runningTasks"]) + len(current_job["finishedTasks"]) == current_job["task_count"]:
                print(f"Job {self.assigned_job} DONE")
                self.update()
                self.assigned_job = None

    def stop(self):
        self.runner_flag = False

    def schedule(self, current_task_id):
        # Multiprocess Robustness
        try:
            # retrieve the necessary data
            job_state, pe_state = self.state.get()
            current_task = self.state.database.get_task_norm(current_task_id)
            input_state = self.get_input(current_task)
        except:
            print("Retrying schedule on : ", self.name)
            self.schedule(current_task_id)

        # First-level scheduling: device selection
        selected_device_index, path = self.local_actor_critic.choose_action(input_state)
        selected_device = self.devices[selected_device_index]
        
        # second-level schedule for non cloud PEs , select a core and a Voltage Frequency Pair
        sub_state = self.get_input(current_task, {0: pe_state[selected_device['id']]},subtree=True)
        sub_state = torch.tensor(sub_state, dtype=torch.float)
        action_logits = self.state.core.forest[selected_device_index](sub_state)
        action_dist = torch.distributions.Categorical(F.softmax(action_logits, dim=-1))
        action = action_dist.sample()
        selected_core_dvfs_index = action.item()
        selected_core_index, dvfs = self._select_core_dvfs(selected_device,selected_core_dvfs_index)

        # applying action on the state and retrieving the result
        reward, fail_flag, energy, time = self.state.apply_action(
            selected_device_index, selected_core_index, dvfs[0], dvfs[1], current_task_id)

        # update the core schudler forest
        self.update_core_scheduler(selected_device_index,action_dist,action,reward)

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
        loss = self.local_actor_critic.calc_loss()  # compute the loss
        self.save_agent_log(loss.item())  # save agent's performance

        self.global_optimizer.zero_grad()  # zero gradients
        loss.backward()

        # Synchronize local and global models
        for local_param, global_param in zip(self.local_actor_critic.parameters(), self.global_actor_critic.parameters()):
            global_param._grad = local_param.grad

        self.global_optimizer.step()  # update global model
        self.local_actor_critic.load_state_dict(self.global_actor_critic.state_dict())  # update local model

    ####### UTILITY FUNCTIONS #######

    def _timeout_on_job(self):
        """Handle timeout when a job is taking too long."""
        print(f"Job {self.assigned_job} TIME OUT")
        self.state.remove_job(self.assigned_job)
        self.assigned_job = None
    def _select_core_dvfs(self, selected_device, selected_core_dvfs_index):
        selected_core_index = int(selected_core_dvfs_index / 3)
        selected_core = selected_device["voltages_frequencies"][selected_core_index]
        dvfs = selected_core[selected_core_dvfs_index % 3]
        return selected_core_index, dvfs

    def get_input(self, task, pe_dict={},device_features=False,subtree=False):
        task_features = [
            task["computational_load"],
            task["input_size"],
            task["output_size"],
            task["kind1"],
            task["kind2"],
            task["kind3"],
            task["kind4"],
            task["is_safe"],
        ]
        if not subtree and not device_features:
            return task_features
        
        pe_features = []
        for pe in pe_dict.values():
            pe_features.extend(self.get_pe_data(pe, pe['id'],subtree))
        return task_features + pe_features

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
