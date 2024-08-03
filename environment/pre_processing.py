from utilities.monitor import Monitor
from data.configs import monitor_config, agent_config


class Preprocessing:
    def __init__(self, state, manager):
        self.state = state
        self.max_jobs = agent_config['multi_agent']
        self.active_jobs = manager.dict()
        self.assigned_jobs = manager.list()
        self.job_pool = manager.dict()
        self.wait_queue = manager.list()
        self.queue = manager.list()

    def run(self):
        jobs, _ = self.state.get()
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
        # TODO : make it faster / maybe : hashmap

        # add window tasks to wait queue
        for task_id in self.state.get_task_window():
            task = self.state.database.get_task(task_id)

            if task['pred_count'] == 0:
                self.queue.append(task_id)
            else:
                self.wait_queue.append(task_id)

        # sort main queue by mobility
        self.__sort_by_mobility()

    def __sort_by_mobility(self):
        mobility_dict = {}
        for task in self.queue:
            mobility_dict[task] = len(
                self.state.database.get_task_successors(task))
        sorted_tasks = list({k: v for k, v in sorted(
            mobility_dict.items(), key=lambda item: item[1])}.keys())
        self.queue[:] = sorted_tasks

    def _is_ready_task(self, task):
        selected_task = self.state.database.get_task(task)
        pred = selected_task['predecessors']
        if selected_task['isReady'] == len(pred):
            return True
        else:
            return False

    def get_agent_queue(self):
        agent_queue = {}
        copy_list = list(self.active_jobs.keys())
        for job_ID in copy_list:
            agent_queue[job_ID] = []
            for task_ID in self.queue:
                task = self.state.database.get_task(task_ID)
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
