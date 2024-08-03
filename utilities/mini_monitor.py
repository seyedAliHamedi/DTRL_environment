import csv
import json
import os
import matplotlib.pyplot as plt
import numpy as np

from data.configs import monitor_config, agent_config, environment_config


class MiniMonitor:

    def __init__(self, n_iterations, manager, config=monitor_config):
        self.env_log = manager.dict()
        self.time_log = manager.dict()
        self.summery = manager.list()
        self.agent_log = manager.dict()
        self._init_time_log(n_iterations, manager)
        self._init_agent_log(manager)

    def _init_time_log(self, n_iterations, manager):
        self.time_log = manager.dict({
            'iterations': n_iterations,
            'time_values': manager.dict(),
            'anomalies': manager.dict(),
            'time_plot': None,
            'avg_iteration_time': -1,
            'cycle-time': environment_config['environment']['cycle'],
        })

    def _init_agent_log(self, manager):
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
        time_values = dict(self.time_log['time_values'])
        y_values = list(time_values.values())
        x_values = list(time_values.keys())

        # average time per iteration
        if y_values:
            avg_iteration_time = sum(y_values) / len(y_values)
        else:
            avg_iteration_time = -1

        total_time = sum(y_values)

        # Update time log dictionary with calculated values
        time_log_dict = {
            'iterations': self.time_log['iterations'],
            'anomalies': dict(self.time_log['anomalies']),
            'time_plot': self.time_log['time_plot'],
            'avg_iteration_time': avg_iteration_time,
            'cycle-time': self.time_log['cycle-time'],
            'total_time': total_time
        }

        # Plotting
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

        # Saving the time log summary
        summary_path = monitor_config['paths']['time']['summery']
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(time_log_dict, f, indent=4)

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
        summary_path = monitor_config['paths']['agent']['summary']
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        self._make_agents_plots(monitor_config['paths']['agent']['plots'])
        with open(summary_path, 'w') as f:
            f.write('Agent-config\n')
            json.dump(agent_config, f, indent=4)
            f.write('\n')
            f.write(f'avg-time: {self.agent_log["summary"]["avg-time"]}\n')
            f.write(f'avg-energy: {self.agent_log["summary"]["avg-energy"]}\n')

    def _save_summery_log(self):
        path = monitor_config['paths']['summary']
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            for i in range(len(self.summery)):
                f.write(f'{self.summery[i]}')

    def add_log(self, log, start='', end='\n'):
        self.summery.append(start)
        self.summery.append(log)
        self.summery.append(end)
