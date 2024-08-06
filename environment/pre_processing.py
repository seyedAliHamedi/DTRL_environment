from utilities.monitor import Monitor
from data.configs import monitor_config, agent_config


class Preprocessing:
    def __init__(self, state, manager, config):
        self.state = state
        self.active_jobs = manager.dict()
        self.assigned_jobs = manager.list()
        self.job_pool = manager.dict()
        self.wait_queue = manager.list()
        self.queue = manager.list()
        self.config = config
        self.max_jobs = config['multi_agent']

    def run(self):
        jobs, _ = self.state.get()
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
        for job_ID in list(state_jobs.keys()):
            self.active_jobs[job_ID] = state_jobs[job_ID]

        # Remove completed jobs from active_jobs
        deleting_list = []
        for job_ID in list(self.active_jobs.keys()):
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
        # add window tasks to wait queue
        for task_id in list(self.state.get_task_window()):
            task = self.state.database.get_task(task_id)

            if task['pred_count'] == 0:
                self.get_queue().append(task_id)
            else:
                self.wait_queue.append(task_id)

        # sort main queue by mobility
        self.__sort_by_mobility()

    def __sort_by_mobility(self):
        mobility_dict = {}
        for task in self.get_queue():
            mobility_dict[task] = len(
                self.state.database.get_task_successors(task))
        sorted_tasks = list({k: v for k, v in sorted(
            mobility_dict.items(), key=lambda item: item[1])}.keys())
        self.get_queue()[:] = sorted_tasks

    def get_queue(self):
        try:
            return self.queue
        except:
            return []

    def get_agent_queue(self):
        agent_queue = {}
        task_list = list(self.get_queue())
        for job_ID in self.active_jobs.keys():
            agent_queue[job_ID] = []
            for task_ID in task_list:
                task = self.state.database.get_task(task_ID)
                if task['job_id'] == job_ID:
                    agent_queue[job_ID].append(task_ID)
        return agent_queue

    def remove_from_queue(self, task_ID):
        self.get_queue().remove(task_ID)
