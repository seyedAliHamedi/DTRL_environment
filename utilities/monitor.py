import csv
import json
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from data.configs import monitor_config, agent_config
from data.configs import environment_config


class Monitor:
    _instance = None

    def __new__(cls, config=monitor_config):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._config = config
            cls._instance.env = None
            cls.env_log = {}
            cls.time_log = {}
            cls.summery = []
            cls.agent_log = {}
        return cls._instance

    def init(self, n_iterations):
        self._init_time_log(n_iterations)
        self._init_main_log()
        self._init_agent_log()

    def _init_time_log(self, n_iterations):
        self.time_log = {
            'iterations': n_iterations,
            'time_values': {},
            'anomalies': {},
            'time_plot': None,
            'avg_iteration_time': -1,
            'cycle-time': environment_config['environment']['cycle'],
        }

    def _init_main_log(self):
        self.env_log['pes'] = {}
        self.env_log['jobs'] = {}
        self.env_log['window'] = {
            'current_cycle': [],
            'pool': [],
            'active_jobs_ID': []
        }
        self.env_log['preprocessing'] = {
            'active_jobs_ID': [],
            'job_pool': [],
            'ready_queue': [],
            'wait_queue': []
        }

    def _init_agent_log(self):
        self.agent_log['live-log'] = {}
        self.agent_log['live-log']['loss'] = []
        self.agent_log['live-log']['reward'] = []
        self.agent_log['live-log']['energy'] = []
        self.agent_log['live-log']['time'] = []
        self.agent_log['live-log']['fail'] = []
        self.agent_log['summary'] = {}

    def add_agent_log(self, log):
        self.agent_log['live-log']['loss'].append(log['loss'])
        self.agent_log['live-log']['reward'].append(log['reward'])
        self.agent_log['live-log']['energy'].append(log['energy'])
        self.agent_log['live-log']['time'].append(log['time'])
        self.agent_log['live-log']['fail'].append(log['fail'])

    def add_time(self, time, iteration):
        if time > environment_config["environment"]["anomaly_th"]:
            self.time_log['anomalies'][iteration] = time
        else:
            self.time_log['time_values'][iteration] = time

    def _save_time_log(self):
        y_values = self.time_log['time_values'].values()
        # average time per iteration
        if len(y_values) != 0:
            self.time_log['avg_iteration_time'] = sum(y_values) / len(y_values)
        else:
            self.time_log['avg_iteration_time'] = -1
        self.time_log['total_time'] = sum(y_values)
        x_values = self.time_log['time_values'].keys()
        del self.time_log['time_values']
        plt.figure(figsize=(10, 5))
        plt.plot(x_values, y_values, marker='o', linestyle='-')
        plt.axhline(y=environment_config['environment']['cycle'],
                    color='red', linestyle='--', label='-set-cycle-time')
        plt.title("Sleeping time on each iteration")
        plt.xlabel("iteration")
        plt.ylabel("sleeping time")
        plt.grid(True)
        plt.legend()
        path = monitor_config['paths']['time']['plot']
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        with open(monitor_config['paths']['time']['summery'], 'w') as f:
            json.dump(self.time_log, f, indent=4)

    def _save_main_log(self):
        main_path = monitor_config['paths']['main']
        os.makedirs(os.path.dirname(main_path['pes']), exist_ok=True)
        os.makedirs(os.path.dirname(main_path['jobs']), exist_ok=True)
        os.makedirs(os.path.dirname(main_path['window']), exist_ok=True)
        os.makedirs(os.path.dirname(main_path['preprocessing']), exist_ok=True)
        self._save_pes_log(main_path['pes'])
        self._save_active_jobs_log(main_path['jobs'])
        self._save_window_log(main_path['window'])
        self._save_preprocessing_log(main_path['preprocessing'])

    def _save_pes_log(self, path):
        with open(path, 'w') as f:
            f.write(pd.DataFrame(self.env_log['pes']).to_string())

    def _save_active_jobs_log(self, path):
        with open(path, 'w') as f:
            f.write(pd.DataFrame(self.env_log['jobs']).to_string())

    def _save_window_log(self, path):
        with open(path, 'w') as f:
            f.write(pd.DataFrame(self.env_log['window']).to_string())

    def _save_preprocessing_log(self, path):
        with open(path, 'w') as f:
            f.write(pd.DataFrame(self.env_log['preprocessing']).to_string())

    def set_env_log(self, state, window_log, preprocessing_log, iteration):
        self.env_log['pes'][iteration] = state[1]
        self.env_log['jobs'][iteration] = state[0]
        self.env_log['window']['pool'].append(window_log['pool'])
        self.env_log['window']['current_cycle'].append(
            window_log['current_cycle'])
        self.env_log['window']['active_jobs_ID'].append(
            window_log['active_jobs_ID'])
        self.env_log['preprocessing']['active_jobs_ID'].append(
            preprocessing_log['active_jobs_ID'])
        self.env_log['preprocessing']['job_pool'].append(
            preprocessing_log['job_pool'])
        self.env_log['preprocessing']['ready_queue'].append(
            preprocessing_log['ready_queue'])
        self.env_log['preprocessing']['wait_queue'].append(
            preprocessing_log['wait_queue'])

    def save_logs(self, time=monitor_config['settings']['time'],
                  main=monitor_config['settings']['main'],
                  summery=monitor_config['settings']['summary'],
                  agent=monitor_config['settings']['agent']):
        if time:
            self._save_time_log()
        if main:
            self._save_main_log()
        if summery:
            self._save_summery_log()
        if agent:
            self._save_agent_log()

    def _save_agent_log(self):
        if len(self.agent_log['live-log']['time']) != 0:
            self.agent_log['summary']['avg-time'] = np.sum(self.agent_log['live-log']['time']) / len(
                self.agent_log['live-log']['time'])
            self.agent_log['summary']['avg-energy'] = np.sum(self.agent_log['live-log']['energy']) / len(
                self.agent_log['live-log']['energy'])
        else:
            self.agent_log['summary']['avg-time'] = -1
            self.agent_log['summary']['avg-energy'] = -1
        self.agent_log['summary']['avg-fail'] = np.sum(
            self.agent_log['live-log']['fail'])
        summary_path = self._config['paths']['agent']['summary']
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        self.__make_agents_plots(self._config['paths']['agent']['plots'])
        with open(summary_path, 'w') as f:
            f.write('Agent-config\n')
            json.dump(agent_config, f, indent=4)
            f.write('\n')
            f.write(f'avg-time: {self.agent_log["summary"]["avg-time"]}\n')
            f.write(f'avg-energy: {self.agent_log["summary"]["avg-energy"]}\n')

    def __make_agents_plots(self, path):
        if len(self.agent_log['live-log']['time']) == 0:
            return
        fig, axs = plt.subplots(3, 2, figsize=(10, 20))
        # Plot each column in a separate plot
        # Plot for loss
        axs[0, 0].plot(self.agent_log['live-log']["loss"],
                       label='Loss', color="blue")
        axs[0, 0].set_title('Loss')
        axs[0, 0].legend()

        # Plot for reward
        axs[0, 1].plot(self.agent_log['live-log']["reward"],
                       label='Reward', color="black")
        axs[0, 1].set_title('Reward')
        axs[0, 1].legend()

        # Plot for time
        axs[1, 0].plot(self.agent_log['live-log']["time"],
                       label='Time', color="red")
        axs[1, 0].set_title('Time')
        axs[1, 0].legend()

        # Plot for energy
        axs[1, 1].plot(self.agent_log['live-log']["energy"],
                       label='Energy', color="green")
        axs[1, 1].set_title('Energy')
        axs[1, 1].legend()

        # Plot for fail
        axs[2, 0].plot(self.agent_log['live-log']["fail"],
                       label='Fail', color="purple")
        axs[2, 0].set_title('Fail')
        axs[2, 0].legend()

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Save the plots to an image file
        plt.savefig(path)

    def _save_summery_log(self):
        path = self._config['paths']['summary']
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            for i in range(len(self.summery)):
                f.write(f'{self.summery[i]}')

    def add_log(self, log, start='', end='\n'):
        self.summery.append(start)
        self.summery.append(log)
        self.summery.append(end)
