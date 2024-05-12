import random

from data.db import Database
from data.configs import environment_config
from environment.state import State


class WindowManager:

    _instance = None

    def __new__(cls,config=environment_config['window']):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.__pool = []
            cls._instance.__head_index = 0
            cls._instance.__max_jobs = config["max_jobs"]
            cls._instance.__window_size = config["size"]
            cls._instance.current_cycle = 1
            cls._instance.__cycle = 3
            cls._instance.active_jobs_ID = []
        return cls._instance

        

    def run(self):
        if self.current_cycle != self.__cycle:
            self.current_cycle += 1
        else:
            self.current_cycle = 0
            State().set_task_window(self.get_window())
        
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

class PreProccesing:
    def __init__(self):
        pass

    @classmethod
    def process(self):
        jobs , _ = State().get()
        
        task_list = []
        for id in jobs.keys():
            for task in jobs[id]['remainingTasks']:
                task_list.append(task)

        return task_list