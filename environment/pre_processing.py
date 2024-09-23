from data.configs import environment_config


class Preprocessing:
    def __init__(self, state, manager, config=environment_config):
        # the shared state
        self.state = state
        # the active jobs of the preprocessor
        self.active_jobs = manager.dict()
        # the assigned jobs to the agents
        self.assigned_jobs = manager.list()
        # the ready 
        self.queue = manager.list()
        
        self.max_jobs = config['multi_agent']
    

    def run(self):
        self.update_active_jobs()
        self.process()

    def assign_job(self):
        # assign jobs to agents
        for job in self.active_jobs.keys():
            if job not in self.assigned_jobs:
                self.assigned_jobs.append(job)
                return job
    def drop_job(self,job_id):
        self.assigned_jobs.remove(job_id)

    def update_active_jobs(self):
        state_jobs = self.state.get_jobs()
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


    def process(self):
        # add window tasks to the queues
        for task_id in self.state.get_task_window():
            task = self.state.database.get_task(task_id)
            # check if the task is ready upon adding it to the queue
            if task['pred_count'] <= 0:
                self.queue.append(task_id)
        # sort main queue by mobility
        self.__sort_by_mobility()

    def __sort_by_mobility(self):
        # sort main queue by mobility
        # mobility -> number of successors(necassity)
        mobility_dict = {}
        for task_id in self.queue:
            task = self.state.database.get_task(task_id)
            mobility_dict[task_id] = len(task['successors'])
        sorted_tasks = list({k: v for k, v in sorted(
            mobility_dict.items(), key=lambda item: item[1])}.keys())
        self.queue[:] = sorted_tasks

    def get_agent_queue(self):
        try:
            queue = self.queue
            job_keys = self.active_jobs.keys()
        except:
            print("Retrying get agent queue")
            return self.get_agent_queue()
        # creating the agent queue dict
        agent_queue = {}
        for job_ID in job_keys:
            agent_queue[job_ID] = []
            for task_ID in queue:
                task = self.state.database.get_task(task_ID)
                if task['job_id'] == job_ID:
                    agent_queue[job_ID].append(task_ID)
        return agent_queue

