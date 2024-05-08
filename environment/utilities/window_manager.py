import random

from data.db import Database
from data.configs import environment_config
from environment.state import State


class WindowManager:

    def __init__(self, config=environment_config["window"]):
        self.__pool = []
        self.__head_index = 0
        self.__max_jobs = config["max_jobs"]
        self.__window_size = config["size"]
        self.current_cycle = 1
        self.__cycle = 3
        self.active_jobs_ID = []

    def get_window(self):
        if self.current_cycle != self.__cycle:
            self.current_cycle += 1
            return []
        self.current_cycle = 0
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
        self.active_jobs_ID.clear()
        for i in range(self.__max_jobs):
            self.active_jobs_ID.append(self.__head_index + i)
        sliced_jobs = Database.get_jobs_window(self.__head_index, self.__max_jobs)
        self.__head_index = self.__head_index + self.__max_jobs
        selected_tasks = []
        for job in sliced_jobs:
            for task in job["tasks_ID"]:
                selected_tasks.append(task)
        random.shuffle(selected_tasks)
        return selected_tasks

    def __update_active_jobs(self):
        jobs_from_list = []
        for job_ID in self.active_jobs_ID:
            jobs_from_list.append(Database.get_job(job_ID))
        State().init_jobs(jobs_from_list)
