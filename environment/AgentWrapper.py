import os

from matplotlib import pyplot as plt

from Agent.Agent import Agent
from data.configs import monitor_config
import torch.optim as optim
from environment.window_manager import Preprocessing
from environment.state import State
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.autograd.set_detect_anomaly(True)


class AgentWrapper:

    def __init__(self, num_devices, devices_features, task_features, db):
        self.db = db
        self.queue = None
        self.actor = Agent(input_size=num_devices * devices_features + task_features, output_size=num_devices)
        self.log = {}

    def run(self, display):
        self.queue = Preprocessing().get_agent_queue()
        if display:
            print(f"Agent  queue: {self.queue}")
        done_tasks = []
        for task in self.queue:
            status = self.schedule(task)
            if status == 0 or True:
                done_tasks.append(task)
        Preprocessing().remove_from_queue(done_tasks)

    def schedule(self, task):
        job_state, pe_state = State().get()
        current_task_id = task
        current_task = self.db.get_task_norm(current_task_id)
        job_id = current_task['job_id']

        state = get_input(self.db, current_task, pe_state)
        state = torch.tensor(state).unsqueeze(0)
        action, probs = self.actor.act(state)

        selected_device_id = action
        selected_device = self.db.get_device(selected_device_id)

        # TODO temp
        selected_core = 0
        selected_freq = 0

        freq = selected_device['voltages_frequencies'][selected_core][selected_freq][0]
        volt = selected_device['voltages_frequencies'][selected_core][selected_freq][1]

        reward, fail_flags, e, t = State().apply_action(action, -1, freq, volt, current_task_id)

        self.add_log(job_id, t, e, reward, fail_flags[0], fail_flags[1], fail_flags[2], fail_flags[3],
                     selected_device_id)

        self.actor.add_experience((state, action, reward))
        self.actor.experience_replay()

        return sum(fail_flags)

    def add_log(self, job_id, time, energy, reward, safe_fail, kind_fail, queue_fail, battery_fail, device_id):
        if job_id not in self.log.keys():
            self.log[job_id] = {
                'time': [],
                'energy': [],
                'reward': [],
                'safe_fail': [],
                'kind_fail': [],
                'queue_fail': [],
                'battery_fail': [],
                'fails': [],
                'iot_usage': [],
                'mec_usage': [],
                'cloud_usage': []
            }
        self.log[job_id]['time'].append(time)
        self.log[job_id]['energy'].append(energy)
        self.log[job_id]['reward'].append(reward)
        self.log[job_id]['fails'].append(safe_fail + kind_fail + queue_fail + battery_fail)
        self.log[job_id]['safe_fail'].append(safe_fail)
        self.log[job_id]['kind_fail'].append(kind_fail)
        self.log[job_id]['queue_fail'].append(queue_fail)
        self.log[job_id]['battery_fail'].append(battery_fail)
        dev_type = self.db.get_device(device_id)['type']
        if dev_type == 'iot':
            self.log[job_id]['iot_usage'].append(1)
            self.log[job_id]['mec_usage'].append(0)
            self.log[job_id]['cloud_usage'].append(0)
        elif dev_type == 'mec':
            self.log[job_id]['iot_usage'].append(0)
            self.log[job_id]['mec_usage'].append(1)
            self.log[job_id]['cloud_usage'].append(0)
        else:
            self.log[job_id]['iot_usage'].append(0)
            self.log[job_id]['mec_usage'].append(0)
            self.log[job_id]['cloud_usage'].append(1)

    def plot_logs(self, save_path):
        time_list = [np.mean(self.log[job_id]['time']) for job_id in self.log]
        energy_list = [np.mean(self.log[job_id]['energy']) for job_id in self.log]
        reward_list = [np.mean(self.log[job_id]['reward']) for job_id in self.log]
        fails_list = [np.mean(self.log[job_id]['fails']) for job_id in self.log]
        safe_fails_list = [np.mean(self.log[job_id]['safe_fail']) for job_id in self.log]
        kind_fails_list = [np.mean(self.log[job_id]['kind_fail']) for job_id in self.log]
        queue_fails_list = [np.mean(self.log[job_id]['queue_fail']) for job_id in self.log]
        battery_fails_list = [np.mean(self.log[job_id]['battery_fail']) for job_id in self.log]
        iot_usage = [np.mean(self.log[job_id]['iot_usage']) for job_id in self.log]
        mec_usage = [np.mean(self.log[job_id]['mec_usage']) for job_id in self.log]
        cc_usage = [np.mean(self.log[job_id]['cloud_usage']) for job_id in self.log]

        fig, axs = plt.subplots(5, 2, figsize=(15, 30))

        # Plot for Loss
        axs[0, 0].plot(reward_list, label='Reward', color="blue", marker='o')
        axs[0, 0].set_title('Reward')
        axs[0, 0].legend()

        # Plot for Time
        axs[0, 1].plot(time_list, label='Time', color="red", marker='o')
        axs[0, 1].set_title('Time')
        axs[0, 1].legend()

        # Plot for Energy
        axs[1, 0].plot(energy_list, label='Energy', color="green", marker='o')
        axs[1, 0].set_title('Energy')
        axs[1, 0].legend()

        # Plot for All Fails
        axs[1, 1].plot(fails_list, label='All Fails', color="purple", marker='o')
        axs[1, 1].set_title('Fails')
        axs[1, 1].legend()

        # Plot for Safe Fails
        axs[2, 0].plot(safe_fails_list, label='Safe Task Fails', color="orange", marker='o')
        axs[2, 0].set_title('Safe Task Fails')
        axs[2, 0].legend()

        # Plot for Kind Fails
        axs[2, 1].plot(kind_fails_list, label='Kind Task Fails', color="brown", marker='o')
        axs[2, 1].set_title('Kind Task Fails')
        axs[2, 1].legend()

        # Plot for Queue Fails
        axs[3, 0].plot(queue_fails_list, label='Queue Full Fails', color="pink", marker='o')
        axs[3, 0].set_title('Queue Full Fails')
        axs[3, 0].legend()

        # Plot for Battery Fails
        axs[3, 1].plot(battery_fails_list, label='Battery Fails', color="cyan", marker='o')
        axs[3, 1].set_title('Battery Fails')
        axs[3, 1].legend()

        # Plot for Device Usage
        axs[4, 0].plot(iot_usage, label='IoT Usage', color='blue', marker='o')
        axs[4, 0].plot(mec_usage, label='MEC Usage', color='orange', marker='x')
        axs[4, 0].plot(cc_usage, label='Cloud Usage', color='green', marker='s')
        axs[4, 0].set_title('Devices Usage History')
        axs[4, 0].set_xlabel('Epochs')
        axs[4, 0].set_ylabel('Usage')
        axs[4, 0].legend()
        axs[4, 0].grid(True)

        # Skip the heatmap plot for now

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Create directories if they do not exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save the plots to an image file
        plt.savefig(save_path)
        plt.close()  # Close the plot to free up memory


####### UTILITY #######
def get_input(db, task, pe_dict):
    task_features = get_task_data(task)
    pe_features = []
    for pe in pe_dict.values():
        pe_features.extend(get_pe_data(db, pe, pe['id']))
    return task_features + pe_features


def get_pe_data(db, pe_dict, pe_id):
    pe = db.get_device(pe_id)
    devicePower = pe['devicePower']

    batteryLevel = pe_dict['batteryLevel']
    battery_capacity = pe['battery_capacity']
    battery_isl = pe['ISL']
    battery = ((1 - battery_isl) * battery_capacity - batteryLevel) / battery_capacity

    cores = sum(pe_dict['occupiedCores'])

    return [cores]


def get_task_data(task):
    return [
        task["computational_load"],
        task["input_size"],
        task["output_size"],
        task["kind1"],
        task["kind2"],
        task["kind3"],
        task["kind4"],
        task["is_safe"],
    ]
