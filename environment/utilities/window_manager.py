import random
from copy import copy

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


class Preprocessing:
    _instance = None

    def __new__(cls, config=environment_config['window']):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.max_jobs = 5
            cls._instance.active_jobs = {}
            cls._instance.job_pool = {}
            cls._instance.wait_queue = []
            cls._instance.queue = []
        return cls._instance

    def process(self):
        # add window tasks to wait queue
        for task in State().get_task_window():
            self.wait_queue.append(task)

        # add ready task from wait queue to main queue
        for task in self.wait_queue:
            if self._is_ready_task(task):
                self.queue.append(task)
                self.wait_queue.remove(task)

        # sort main queue by mobility
        self.__sort_by_mobility()

    def __sort_by_mobility(self):
        mobility_dict = {}
        for task in self.queue:
            mobility_dict[task] = len(Database.get_task_successors(task))
        self.queue = list({k: v for k, v in sorted(mobility_dict.items(), key=lambda item: item[1])}.keys())

    def _is_ready_task(self, task):
        selected_task = Database.get_task(task)
        task_job_id = selected_task['job_id']
        if task_job_id not in self.active_jobs:
            return False
        state_job = self.active_jobs[task_job_id]
        task_pred = copy(selected_task['predecessors'])
        for task in state_job['finishedTasks']:
            if task in task_pred:
                task_pred.remove(task)
        if len(task_pred) == 0:
            return True
        else:
            return False

    def update_active_jobs(self, state_jobs):
        # add to active jobs
        for job_ID in state_jobs.keys():
            if job_ID not in self.active_jobs.keys():
                self.active_jobs[job_ID] = state_jobs[job_ID]
            else:
                # update job values from state
                self.active_jobs[job_ID] = state_jobs[job_ID]

        # remove from active jobs
        deleting_list = []
        for job_ID in self.active_jobs.keys():
            job = self.active_jobs[job_ID]
            if len(job['finishedTasks']) + len(job["runningTasks"]) == job["task_count"]:
                deleting_list.append(job_ID)
        for item in deleting_list:
            self.active_jobs.pop(item)

        # add to job_pool
        while len(self.active_jobs.keys()) > self.max_jobs:
            job_ID, job = self.active_jobs.popitem()
            self.job_pool[job_ID] = job

        # remove from job_pool
        while (len(self.active_jobs.keys()) < self.max_jobs) and len(self.job_pool.keys()) > 0:
            job_ID, job = self.job_pool.popitem()
            self.active_jobs[job_ID] = job

    def run(self):
        jobs, _ = State().get()
        self.update_active_jobs(jobs)
        self.process()
        print(f"active_jobs: {self.active_jobs.keys()}")
        print(f"job_pool: {self.job_pool.keys()}")
        print(f"wait_queue: {self.wait_queue}")
        print(f"queue: {self.queue}")
        State().set_agent_queue(self.queue)
