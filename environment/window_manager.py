import random

from data.db import Database
from data.configs import environment_config

class WindowManager:

    def __init__(self, jobs, config=environment_config["window"]):
        self._pool = []
        self.__head_index = 0
        self.__max_jobs = config["max_jobs"]
        self.__current_window_size = -1

    def get_window(self):
        window = []
        if len(self._pool) == 0:
            self.__pool ,task_count= self.__slice()
            self.__current_window_size = task_count / self.__max_jobs
            print(f"len(selected_tasks) = {len(self.__current_task_pool)}")
            print(f"window_counts = {self.__window_counts}")
            print(f"window_size = {self.__window_size}")
        else:
            for i in range(self.__current_window_size):
                window.append(self._pool.pop(0))
        return window

    def __slice(self):
        pool = Database.get_jobs_window(self.__head_index,self.__max_jobs)
        self.__head_index = self.__head_index + self.__max_jobs
        selected_tasks = []
        task_count = 0
        for job in pool:
            task_count += len(job["tasks_ID"])
            for task in job["tasks_ID"]:
                selected_tasks.append(task)
        random.shuffle(selected_tasks)
        return selected_tasks,task_count


