import json
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from data.configs import environment_config, monitor_config


class Monitor:

    def __init__(self):
        self.db = MonitorDataBase()

    def log_PEs(self, PEs_dict):
        for pe in PEs_dict.values():
            avg_usage = np.mean(pe['occupiedCores'])
            self.db.add_PE_log(f'{pe["type"][0]}{pe["id"]}', avg_usage)

    def plot(self, file_path='./logs/simulation/'):
        pe_log_dict = {}
        for pe_name, values in self.db.PEs.items():
            pe_log_dict[pe_name] = np.mean(values[-100:])

        # Bar Plot: Activity values for each processing element
        plt.figure(figsize=(8, 5))
        plt.bar(pe_log_dict.keys(), pe_log_dict.values(), color='skyblue')

        plt.xlabel('Processing Element')
        plt.ylabel('Activity')
        plt.title('Activity for Each Processing Element')

        plt.savefig(file_path + 'pes_activity', dpi=300)
        plt.close()


class MonitorDataBase:
    def __init__(self):
        self.PEs = {}

    def add_PE_log(self, pe_name, pe_log):
        try:
            self.PEs[pe_name].append(pe_log)
        except KeyError:
            self.PEs[pe_name] = []
            self.PEs[pe_name].append(pe_log)
