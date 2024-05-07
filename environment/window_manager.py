import random

from data.db import Database
from data.configs import environment_config

class WindowManager:

    def __init__(self, config=environment_config["window"]):
        self.__pool = []
        self.__head_index = 0
        self.__max_jobs = config["max_jobs"]
        self.__window_size = config["size"]

    def get_window(self):
        window = []
        if len(self.__pool) == 0:
            self.__pool = self.__slice()

            print(f"pool  = {self.__pool}")
            print(f"len(selected_tasks) = {len(self.__pool)}")
            print(f"window_size = {self.__window_size}")

        elif len(self.__pool)<self.__window_size:
            for i in range(len(self.__pool)):
                window.append(self.__pool.pop(0))
            return window
        else:
            for i in range(self.__window_size):
                window.append(self.__pool.pop(0))
        return window

    def __slice(self):
        sliced_jobs = Database.get_jobs_window(self.__head_index,self.__max_jobs)
        self.__head_index = self.__head_index + self.__max_jobs
        selected_tasks = []
        for job in sliced_jobs:
            for task in job["tasks_ID"]:
                selected_tasks.append(task)
        random.shuffle(selected_tasks)
        return selected_tasks
