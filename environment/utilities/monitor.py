import json
import os

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
        return cls._instance

    def init(self, n_iterations):
        self._init_time_log(n_iterations)
        self._init_main_log(n_iterations)

    def _init_time_log(self, n_iterations):
        self.time_log = {
            'iterations': n_iterations,
            'time_values': {},
            'anomalies': {},
            'time_plot': None,
            'avg_iteration_time': -1,
            'cycle-time': environment_config['environment']['cycle'],
        }

    def _init_main_log(self, n_iterations):
        pass

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
        plt.figure(figsize=(10, 5))  # Optional: Set the figure size
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
        main_path = monitor_config['path']['main']
        os.makedirs(os.path.dirname(main_path['pes']), exist_ok=True)
        os.makedirs(os.path.dirname(main_path['jobs']), exist_ok=True)
        os.makedirs(os.path.dirname(main_path['window']), exist_ok=True)
        os.makedirs(os.path.dirname(main_path['preprocessing']), exist_ok=True)

    def _save_pes_log(self, path, pe_ID, iteration):
        with open(path, 'w') as f:
            json.dump(self.env_log['pes'][iteration]['pe_ID'], f, indent=4)



    def save_logs(self, time=True, main=True):
        if time:
            self._save_time_log()
        if main:
            self._save_main_log()
