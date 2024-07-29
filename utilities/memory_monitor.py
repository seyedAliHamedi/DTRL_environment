import threading
import time

import psutil
from matplotlib import pyplot as plt
from data.configs import monitor_config


class MemoryMonitor:
    def __init__(self, interval=1, max_mem_usage=2000, warn=False):
        self.interval = interval
        self._stop_event = threading.Event()
        self._max_mem_usage = max_mem_usage
        self._mem_log = []
        self._warn = warn
        self._running = False

    def stop(self):
        self._running = False
        self.save_memory_usage()

    def run(self):
        self._running = True
        max_mem_usage = self._max_mem_usage
        while self._running:
            memory_usage = self.get_memory_usage()
            self._mem_log.append(memory_usage)
            if memory_usage > max_mem_usage and self._warn:
                print(f"\033[91mMemory usage: {memory_usage} MB\033[0m")
            time.sleep(self.interval)

    def save_memory_usage(self):
        fig, axis = plt.subplots()
        axis.plot(self._mem_log, color="r", label="Memory usage")
        axis.set_title("Memory usage")
        axis.set_xlabel("Time")
        axis.set_ylabel("Memory usage")
        axis.legend()
        plt.savefig(monitor_config["paths"]["memory"])

    def get_memory_usage(self):
        process = psutil.Process()
        return process.memory_info().rss / (1024 ** 2)  # Memory usage in megabytes
