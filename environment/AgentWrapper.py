from Agent.Agent import Agent
from data.configs import monitor_config
import torch.optim as optim
from environment.window_manager import Preprocessing
from environment.state import State
from data.db import Database
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.autograd.set_detect_anomaly(True)


class AgentWrapper:

    def __init__(self, num_devices, devices_features, task_features):
        self.queue = None
        self.actor = Agent(input_size=num_devices * devices_features + task_features, output_size=num_devices)
        self.log = {}

    def run(self, display):
        self.queue = Preprocessing().get_agent_queue()
        if display:
            print(f"Agent  queue: {self.queue}")
        for task in self.queue:
            self.schedule(task)

    def schedule(self, task):
        job_state, pe_state = State().get()
        current_task_id = task
        current_task = Database().get_task(current_task_id)
        job_id = current_task['job_id']

        state = get_input(current_task, pe_state)
        state = torch.tensor(state).unsqueeze(0)
        action, probs = self.actor.act(state)

        selected_device_id = action
        selected_device = Database.get_device(selected_device_id)

        # TODO temp
        selected_core = 0
        selected_freq = 0

        freq = selected_device['voltages_frequencies'][selected_core][selected_freq][0]
        volt = selected_device['voltages_frequencies'][selected_core][selected_freq][1]

        reward, fail_flags, e, t = State().apply_action(action, -1, freq, volt, current_task_id)

        self.log[job_id]['time'] = t
        self.log[job_id]['energy'] = e
        self.log[job_id]['reward'] = reward
        self.log[job_id]['safe_fail'] = fail_flags[0]
        self.log[job_id]['kind_fail'] = fail_flags[1]
        self.log[job_id]['queue_fail'] = fail_flags[2]
        self.log[job_id]['battery_fail'] = 0


        self.actor.add_experience((state, action, reward))
        self.actor.experience_replay()

    def add_log(self, job_id, time, energy, reward, safe_fail, kind_fail, queue_fail, battery_fail, device_id):
        try:
            self.log[job_id]


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

    return [cores, devicePower, battery]
