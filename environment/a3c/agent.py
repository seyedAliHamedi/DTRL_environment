import threading
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

from data.db import Database
from environment.a3c.actor_critic import ActorCritic
from environment.a3c.shared_adam import SharedAdam
from environment.dtrl.core_scheduler import CoreScheduler
from environment.state import State
from environment.window_manager import Preprocessing
from utilities.monitor import Monitor
from data.configs import monitor_config


class Agent(mp.Process):
    def __init__(self, name, global_actor_critic, optimizer, barrier):
        super(Agent, self).__init__()
        self.global_actor_critic = global_actor_critic
        self.local_actor_critic = ActorCritic(
            self.global_actor_critic.input_dims, self.global_actor_critic.n_actions)
        self.optimizer = optimizer
        self.name = name
        self.num_input = 5
        self.num_output = 5
        self.devices = Database.get_all_devices()
        self.assigned_job = None
        self.task_queue = []
        self.core = CoreScheduler(self.devices)
        # self.thread = threading.Thread(target=self.run, name=self.name)
        # Shared boolean flag for stopping
        self.runner_flag = mp.Value('b', True)
        self.barrier = barrier

    def run(self):
        while self.runner_flag:
            if self.assigned_job is None:
                self.assigned_job = Preprocessing().assign_job()
                if self.assigned_job is None:
                    self.barrier.wait()
                    continue
                self.local_actor_critic.clear_memory()
            self.task_queue = Preprocessing().get_agent_queue()[
                self.assigned_job]
            self.schedule()
            current_job = State().get_job(self.assigned_job)

            if len(current_job["runningTasks"]) + len(current_job["finishedTasks"]) == current_job["task_count"]:
                State().jobs_done += 1
                total_loss = self.update()
                self.assigned_job = None
                if monitor_config['settings']['agent']:
                    Monitor().add_agent_log(
                        {
                            'loss': total_loss.item(),
                            #     'reward': sum(self.rewards[self.assigned_job]) / len(self.rewards[self.assigned_job]),
                            #     'time': sum(self.time[self.assigned_job]) / len(self.time[self.assigned_job]),
                            #     'energy': sum(self.energy[self.assigned_job]) / len(self.energy[self.assigned_job]),
                            #     'fail': sum(self.fail[self.assigned_job]) / len(self.fail[self.assigned_job]),
                        }
                    )
            self.barrier.wait()

    def stop(self):
        self.runner_flag = False

    def schedule(self):
        if len(self.task_queue) == 0:
            return
        job_state, pe_state = State().get()
        current_task_id = self.task_queue[0]
        current_task = Database().get_task(current_task_id)
        current_job = State().get_job(self.assigned_job)
        input_state = get_input(current_task, {})

        option = self.local_actor_critic.choose_action(input_state)

        selected_device_index = option
        selected_device = self.devices[option]

        if selected_device['type'] != "cloud":
            sub_state = get_input(
                current_task, {0: pe_state[selected_device['id']]})
            sub_state = torch.tensor(sub_state, dtype=torch.float32)
            action_logits = self.core.forest[selected_device_index](sub_state)
            action_dist = torch.distributions.Categorical(
                F.softmax(action_logits, dim=-1))
            action = action_dist.sample()
            selected_core_index = action.item()
            selected_core = selected_device["voltages_frequencies"][selected_core_index]
            dvfs = selected_core[np.random.randint(0, 3)]

        if selected_device['type'] == "cloud":
            selected_core_index = -1
            i = np.random.randint(0, 1)
            dvfs = [(50000, 13.85), (80000, 24.28)][i]

        if monitor_config['settings']['agent']:
            Monitor().add_log(
                f"Agent Action::Device: {selected_device_index} |"
                f" Core: {selected_core_index} | freq: {dvfs[0]} |"
                f" vol: {dvfs[1]} | task_id: {current_task_id} |"
                f" cl: {Database.get_task(current_task_id)['computational_load']}", start='\n', end='')

        reward, fail_flag, energy, time = State().apply_action(selected_device_index,
                                                               selected_core_index, dvfs[0], dvfs[1], current_task_id)

        self.local_actor_critic.archive(input_state, option, reward)

        if fail_flag == 0:
            Preprocessing().remove_from_queue(current_task_id)
            self.task_queue = Preprocessing().get_agent_queue()[
                self.assigned_job]

    def update(self):
        loss = self.local_actor_critic.calc_loss()
        loss.backward()
        for local_param, global_param in zip(self.local_actor_critic.parameters(),
                                             self.global_actor_critic.parameters()):
            global_param._grad = local_param.grad
        self.optimizer.step()
        self.local_actor_critic.load_state_dict(
            self.global_actor_critic.state_dict())
        self.local_actor_critic.clear_memory()

        return loss


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
