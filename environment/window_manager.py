import numpy as np
from data.configs import environment_config
class WindowManager:

    def __init__(self, jobs, config=environment_config["window"]):
        self.__jobs_list = jobs
        self.__jobs_pool = None
        self.__window_size = config["size"]
        self.__max_jobs_in_slice = config["max_jobs"]
        self.__current_task_pool = []
        self.__head_job_index = 0
        self.__tail_job_index = 0

    def get_window(self):
        window = []
        if len(self.__current_task_pool) == 0:
            self.__current_task_pool = self.__slice()
            if len(self.__current_task_pool) < self.__window_size:
                for i in range(len(self.__current_task_pool)):
                    window.append(self.__current_task_pool.pop())
            else:
                for i in range(self.__window_size):
                    window.append(self.__current_task_pool.pop())
        elif len(self.__current_task_pool) <= self.__window_size:
            for _ in range(len(self.__current_task_pool)):
                window.append(self.__current_task_pool.pop())
        else:
            for i in range(self.__window_size):
                window.append(self.__current_task_pool.pop())
        return window

    def __slice(self):
        self.__tail_job_index = self.__head_job_index + self.__max_jobs_in_slice
        self.__jobs_pool = self.__jobs_list[self.__head_job_index:self.__tail_job_index]
        print(f"sliced for {self.__head_job_index},{self.__tail_job_index}")
        self.__head_job_index = self.__tail_job_index
        selected_tasks = Helper.all_tasks(self.__jobs_pool)
        np.random.shuffle(np.array(selected_tasks))
        return selected_tasks

    def get_slice_range(self):
        return self.__head_job_index, self.__tail_job_index


jobs = JobGenerator().generate(20)[0]
window_manager = WindowManager(jobs)
window_buffer = []

for i in range(40):
    print(f"$$$$$$$$$$$$$$$$$$$$$$$$$$$$//{i}//")
    window_buffer = window_manager.get_window()
    for job in window_buffer:
        job.info()
    time.sleep(1)