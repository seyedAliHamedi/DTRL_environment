import json
import os

import pandas as pd
from matplotlib import pyplot as plt

from data.configs import monitor_config
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
        return cls._instance

    def init(self, n_iterations):
        self._init_time_log(n_iterations)
        self._init_main_log()

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
        self.env_log['window'] = {}
        self.env_log['preprocessing'] = {}

    def add_env_log(self, iteration, key, new_content):
        self.env_log[key][iteration] = new_content

    def add_time(self, time, iteration):
        if time > 0.5:
            self.time_log['anomalies'][iteration] = time
        else:
            self.time_log['time_values'][iteration] = time

    def _save_time_log(self):
        y_values = self.time_log['time_values'].values()
        # average time per iteration
        self.time_log['avg_iteration_time'] = sum(y_values) / len(y_values)
        x_values = self.time_log['time_values'].keys()
        plt.figure(figsize=(10, 5))
        plt.plot(x_values, y_values, marker='o', linestyle='-')
        plt.axhline(y=environment_config['environment']['cycle'], color='red', linestyle='--', label='-set-cycle-time')
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
            # json.dump(self.env_log['pes'], f, indent=4)

    def _save_active_jobs_log(self, path):
        with open(path, 'w') as f:
            f.write(pd.DataFrame(self.env_log['jobs']).to_string())
            # json.dump(self.env_log['jobs'], f, indent=4)

    def _save_window_log(self, path):
        with open(path, 'w') as f:
            f.write(pd.DataFrame(self.env_log['window']).to_string())

    def _save_preprocessing_log(self, path):
        with open(path, 'w') as f:
            f.write(pd.DataFrame(self.env_log['preprocessing']).to_string())

    def set_env_log(self, state, window_log, preprocessing_log, iteration):
        self.env_log['pes'][iteration] = state[1]
        self.env_log['jobs'][iteration] = state[0]
        self.env_log['window'][iteration] = window_log
        self.env_log['preprocessing'][iteration] = preprocessing_log

    def save_logs(self, time=True, main=True, summery=True):
        if time:
            self._save_time_log()
        if main:
            self._save_main_log()
        if summery:
            self._save_summery_log()

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
