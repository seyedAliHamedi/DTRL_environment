from data.configs import environment_config


class Preprocessing:
    def __init__(self, state, manager, config=environment_config):
        self.state = state
        self.active_jobs = manager.dict()
        self.assigned_jobs = manager.list()
        self.job_pool = manager.dict()
        # the ready & wait queue
        self.wait_queue = manager.list()
        self.queue = manager.list()
        self.max_jobs = config['multi_agent']

    def run(self):
        jobs = self.state.get_jobs()
        self.update_active_jobs(jobs)
        self.process()

    def assign_job(self):
        with self.state.lock:
            for job in self.active_jobs.keys():
                if job not in self.assigned_jobs:
                    self.assigned_jobs.append(job)
                    return job

    def update_active_jobs(self, state_jobs):
        # Add or update jobs in active_jobs
        for job_ID in state_jobs.keys():
            self.active_jobs[job_ID] = state_jobs[job_ID]

        # Remove completed jobs from active_jobs if they are all scheduled
        deleting_list = []
        for job_ID in self.active_jobs.keys():
            job = self.active_jobs[job_ID]
            if len(job['finishedTasks']) + len(job["runningTasks"]) == job["task_count"]:
                deleting_list.append(job_ID)

        for job_ID in deleting_list:
            self.active_jobs.pop(job_ID)

        # Move excess jobs from active_jobs to job_pool
        while len(self.active_jobs) > self.max_jobs:
            job_ID, job = self.active_jobs.popitem()
            self.job_pool[job_ID] = job

        # Move jobs from job_pool to active_jobs if there's space
        while len(self.active_jobs) < self.max_jobs and len(self.job_pool) > 0:
            job_ID, job = self.job_pool.popitem()
            self.active_jobs[job_ID] = job

    def process(self):
        # add window tasks to the queues
        for task_id in self.state.get_task_window():
            task = self.state.database.get_task(task_id)
            # check if the task is ready upon adding it to the queue
            if task['pred_count'] == 0:
                self.queue.append(task_id)
            else:
                self.wait_queue.append(task_id)
        # sort main queue by mobility
        self.__sort_by_mobility()

    def __sort_by_mobility(self):
        mobility_dict = {}
        for task_id in self.queue:
            task = self.state.database.get_task(task_id)
            mobility_dict[task_id] = len(task['successors'])
        sorted_tasks = list({k: v for k, v in sorted(
            mobility_dict.items(), key=lambda item: item[1])}.keys())
        self.queue[:] = sorted_tasks

    def get_agent_queue(self):
        # creating the agent queue dict
        agent_queue = {}
        for job_ID in self.active_jobs.keys():
            agent_queue[job_ID] = []
            for task_ID in list(self.queue):
                task = self.state.database.get_task(task_ID)
                if task['job_id'] == job_ID:
                    agent_queue[job_ID].append(task_ID)
        return agent_queue

