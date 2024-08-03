import random
from data.db import Database
from data.configs import environment_config, monitor_config, agent_config
from environment.state import State


class WindowManager:

    def __init__(self, env, config=environment_config['window']):
        self.__pool = []
        self.__head_index = 0
        self.__max_jobs = config["max_jobs"]
        self.__window_size = config['size']
        self.__cycle = config['clock']
        self.active_jobs_ID = []
        self.current_cycle = config["clock"]
        self.env = env

    def run(self):
        if self.current_cycle != self.__cycle:
            self.current_cycle += 1
            self.env.state.set_task_window([])
        else:
            self.current_cycle = 0
            self.env.state.set_task_window(self.get_window())

    def get_window(self):
        window = []
        if len(self.__pool) == 0:
            self.__pool = self.__slice()

        if len(self.__pool) < self.__window_size:
            for i in range(len(self.__pool)):
                window.append(self.__pool.pop(0))
            return window
        else:
            for i in range(self.__window_size):
                window.append(self.__pool.pop(0))
        return window

    def __slice(self):
        sliced_jobs = Database.get_jobs_window(
            self.__head_index, self.__max_jobs)
        self.__head_index = self.__head_index + self.__max_jobs
        selected_tasks = []
        for job in sliced_jobs:
            for task in job["tasks_ID"]:
                selected_tasks.append(task)
        random.shuffle(selected_tasks)
        return selected_tasks

    def get_log(self):
        return {
            'pool': self.__pool,
            'current_cycle': self.current_cycle,
            'active_jobs_ID': self.active_jobs_ID,
        }
