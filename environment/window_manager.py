import random
import time

import numpy as np
from data.configs import environment_config
from data.db import Database
from data.gen import Generator


class WindowManager:

    def __init__(self, jobs, config=environment_config["window"]):
        self.__jobs_list = jobs
        self.__jobs_pool = None
        self.__window_size = config["size"]
        self.__window_counts = -1
        self.__max_jobs_in_slice = config["max_jobs"]
        self.__current_task_pool = []
        self.__head_job_index = 0
        self.__tail_job_index = 0

    def get_window(self):
        window = []
        if len(self.__current_task_pool) == 0:
            self.__current_task_pool = self.__slice()
            self.__window_counts = len(self.__current_task_pool) / self.__window_size
            if self.__window_counts > float(int(self.__window_counts)):
                self.__window_counts = int(self.__window_counts) + 1
            print(f"len(selected_tasks) = {len(self.__current_task_pool)}")
            print(f"window_counts = {self.__window_counts}")
            print(f"window_size = {self.__window_size}")
        if len(self.__current_task_pool) < self.__window_size:
            for i in range(len(self.__current_task_pool)):
                window.append(self.__current_task_pool.pop(0))
        else:
            for i in range(self.__window_size):
                window.append(self.__current_task_pool.pop(0))
        return window

    def __slice(self):
        self.__tail_job_index = self.__head_job_index + self.__max_jobs_in_slice
        self.__jobs_pool = self.__jobs_list[self.__head_job_index:self.__tail_job_index]
        self.__head_job_index = self.__tail_job_index
        selected_tasks = []
        for job in self.__jobs_pool:
            for task in job["tasks_ID"]:
                selected_tasks.append(task)
        random.shuffle(selected_tasks)
        return selected_tasks

    def get_slice_range(self):
        return self.__head_job_index, self.__tail_job_index


Database.load()
window_manager = WindowManager(Database.get_all_jobs())
while True:
    starting_time = time.time()
    buffer = window_manager.get_window()
    ending_time = time.time()
    print(buffer)
    print(f"total_time = {ending_time - starting_time}\n")
    time.sleep(1)