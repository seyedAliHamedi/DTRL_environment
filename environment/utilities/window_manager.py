import random

from data.db import Database
from data.configs import environment_config
from environment.state import State


class WindowManager:
    _instance = None

    def __new__(cls, config=environment_config['window']):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.__pool = []
            cls._instance.__head_index = 0
            cls._instance.__max_jobs = config["max_jobs"]
            cls._instance.__window_size = config["size"]
            cls._instance.current_cycle = 1
            cls._instance.__cycle = config["clock"]
            cls._instance.active_jobs_ID = []
        return cls._instance

    def run(self):
        if self.current_cycle != self.__cycle:
            self.current_cycle += 1
            State().set_task_window([])
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
        sliced_jobs = Database.get_jobs_window(self.__head_index, self.__max_jobs)
        self.__head_index = self.__head_index + self.__max_jobs
        selected_tasks = []
        for job in sliced_jobs:
            for task in job["tasks_ID"]:
                selected_tasks.append(task)
        random.shuffle(selected_tasks)
        return selected_tasks



class PreProccesing:
    def __init__(self):
        self.max_jobs = 5
        self.active_jobs_id = []
        self.job_pool = []
        

        ####################
        self.wait_queue = []
        self.agent_queue = []


    def process(self):
        for job in self.active_jobs_id:
            for task in job['remainingTasks']:
                if task not in self.agent_queue:
                    #TODO : preprocess             | amin
                    self.agent_queue.append(task)

    def update_active_jobs(self,state_jobs):
        for job in state_jobs :
            if job not in self.active_jobs_id:
                self.active_jobs_id.append(State().get_job(job))

        while len(self.active_jobs_id)>self.max_jobs:
            job = self.active_jobs_id.pop()
            self.job_pool.append(job)

        while len(self.active_jobs_id)<self.max_jobs and len(self.job_pool)>0:
            job = self.job_pool.pop(0)
            self.active_jobs_id.append(job)


    def run(self):
        jobs, _ = State().get()
        self.update_active_jobs(jobs)
        self.process()
        State().set_agent_queue(self.agent_queue)
