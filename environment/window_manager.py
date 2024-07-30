import random
from copy import copy

from data.db import Database
from data.configs import environment_config, monitor_config,agent_config
from environment.state import State
from utilities.monitor import Monitor


class WindowManager:
    _instance = None

    def __new__(cls, config=environment_config['window']):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.__pool = []
            cls._instance.__head_index = 0
            cls._instance.__max_jobs = config["max_jobs"]
            cls._instance.__window_size = config["size"]
            cls._instance.current_cycle = config["clock"]
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
            'pool': self._instance.__pool,
            'current_cycle': self._instance.current_cycle,
            'active_jobs_ID': self._instance.active_jobs_ID,
        }


class Preprocessing:
    _instance = None

    def __new__(cls, config=environment_config['window']):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.max_jobs = agent_config['multi_agent']
            cls._instance.active_jobs = {}
            cls._instance.assigned_jobs = []
            cls._instance.job_pool = {}
            cls._instance.wait_queue = []
            cls._instance.queue = []
        return cls._instance

    def run(self):
        jobs, _ = State().get()
        self.update_active_jobs(jobs)
        self.process()

        if monitor_config['settings']['main']:
            Monitor().add_log(f"active_jobs: {self.active_jobs.keys()}")
            Monitor().add_log(f"job_pool: {self.job_pool.keys()}")
            Monitor().add_log(f"wait_queue: {self.wait_queue}")
            Monitor().add_log(f"queue: {self.queue}")

    def assign_job(self):
        for job in self.active_jobs.keys():
            if job not in self.assigned_jobs:
                self.assigned_jobs.append(job)
                return job

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

    def process(self):
        #TODO : make it faster / maybe : hashmap

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
        self.queue = list({k: v for k, v in sorted(
            mobility_dict.items(), key=lambda item: item[1])}.keys())

    def _is_ready_task(self, task):
        selected_task = Database.get_task(task)
        pred = selected_task['predecessors']
        if selected_task['isReady'] == len(pred):
            return True
        else:
            return False

    def get_agent_queue(self):
        agent_queue = {}
        copy_list = copy(list(self.active_jobs.keys()))
        for job_ID in copy_list:
            agent_queue[job_ID] = []
            for task_ID in self.queue:
                task = Database().get_task(task_ID)
                if task['job_id'] == job_ID:
                    agent_queue[job_ID].append(task_ID)
        return agent_queue

    def remove_from_queue(self, task_ID):
        try:
            self.queue.remove(task_ID)
        except ValueError:
            raise f"task{task_ID} is not in window-manager queue"

    def get_log(self):
        return {
            'active_jobs_ID': list(self.active_jobs.keys()),
            'job_pool': list(self.job_pool.keys()),
            'ready_queue': self.queue,
            'wait_queue': self.wait_queue,
        }
