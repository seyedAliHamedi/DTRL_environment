import random
from data.configs import environment_config

class WindowManager:

    def __init__(self, state, manager, config=environment_config):
        # the shared state
        self.state = state
        self.__pool = manager.list()
        # the active jobs in the window manager
        self.active_jobs_ID = manager.list()
        # max jobs in a window & window size
        self.__max_jobs = config['window']["max_jobs"]
        self.__window_size = config['window']["size"]
        # cycles to handle the jobs generations process
        self.current_cycle = config['window']["clock"]
        self.__cycle = config['window']["clock"]
        # head index to keep track of jobs read from db
        self.__head_index = 1


    def run(self):
        # pass the windows to state every defined cycle
        if self.current_cycle != self.__cycle:
            self.current_cycle += 1
            self.state.set_task_window([])
        else:
            if len(self.state.get_jobs())>30:
                self.current_cycle -=1
            else:
                self.current_cycle = 0
                self.state.set_task_window(self.get_window())

    def get_window(self):
        # return the entire pool or a protion of it(window size)
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
        # get the nex max jobs and pour them into the pool
        sliced_jobs = self.state.database.get_jobs_window(
            self.__head_index, self.__max_jobs)
        self.__head_index = self.__head_index + self.__max_jobs
        selected_tasks = []
        for job in sliced_jobs:
            for task in job["tasks_ID"]:
                selected_tasks.append(task)
        random.shuffle(selected_tasks)
        return selected_tasks
