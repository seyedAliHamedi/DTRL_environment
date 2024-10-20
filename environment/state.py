import numpy as np
import pandas as pd
import torch.multiprocessing as mp



from environment.util import *
from environment.pre_processing import Preprocessing
from environment.window_manager import WindowManager

from configs import environment_config


class State:

    def __init__(self, devices ,jobs ,tasks, manager):
        self.db_devices = devices
        self.db_jobs = jobs
        self.db_tasks = {item['id']: item for item in tasks}
        
        # the state live values PEs & Jobs
        self.PEs = manager.list()
        self.jobs = manager.dict()
        
        # initializing PEs in Idle from database
        self.init_PEs(self.db_devices, manager)
        
        # the task window manged by the window manager
        self.task_window = manager.list()
        
        # initializing the preprocessor and the window manager
        self.preprocessor = Preprocessing(state=self, manager=manager)
        self.window_manager = WindowManager(state=self, manager=manager)

        # TODO : rest
        self.agent_log = manager.dict({})
        self.paths = manager.list([])
        self.display = environment_config['display']
        self.lock = mp.Lock()
        

    ##### Intializations
    def init_PEs(self, PEs, manager):
        # initializing the PEs live variable from db
        for pe in PEs:
            self.PEs.append({
                "type": pe["type"],
                "batteryLevel": pe["battery_capacity"],
                "occupiedCores": manager.list([0 for _ in range(pe["num_cores"])]),
                "energyConsumption": manager.list([pe["powerIdle"] for _ in range(pe["num_cores"])]),
                "queue": manager.list(
                    [manager.list([(0, -1) for _ in range(pe["maxQueue"])]) for _ in range(pe["num_cores"])]),
            })

    def set_jobs(self, jobs, manager):
        # add new job the state live status (from window manager)
        for job in jobs:
            self.jobs[job["id"]] = {
                "id": job["id"],
                "task_count": job["task_count"],
                "finishedTasks": manager.list([]),
                "runningTasks": manager.list([]),
                "remainingTasks": manager.list([]),
                "remainingDeadline": job["deadline"],
            }

    ##### Functionality
    def apply_action(self, pe_ID, core_i, freq, volt, task_ID):
        # Mutliprocessing robustness
        try:
            pe_dict = self.PEs[pe_ID]
            pe = self.db_devices[pe_ID]
            task = self.db_tasks[task_ID]
            job_dict = self.jobs[task["job_id"]]
        except:
            print("Retry apply action")
            return self.apply_action(pe_ID, core_i, freq, volt, task_ID)
        
        
        with self.lock:
            execution_time = np.ceil(task["computational_load"] / freq)

            if execution_time > 5:
                execution_time = 5
            # TODO t must include time of tasks scheduled before it ,in selected queue
            placing_slot = (execution_time, task_ID)

            queue_index, core_index, lag_time = self.find_place(pe_dict, core_i)

            fail_flags = [0, 0, 0, 0]
            if task["is_safe"] and not pe['is_safe']:
                # fail : assigned safe task to unsafe device
                fail_flags[0] = 1
            if task["task_kind"] not in pe["acceptable_tasks"]:
                # fail : assigned a kind of task to the inappropriate device
                fail_flags[1] = 1
            if queue_index == -1 and core_index == -1:
                # fail : assigned a task to a full queue core
                fail_flags[2] = 0

            if sum(fail_flags) > 0:
                return sum(fail_flags) * reward_function(punish=True), fail_flags, 0, 0

            # print(f'task{task_ID},   device{pe_ID},   core{core_index},    queue{queue_index}')
            pe_dict["queue"][core_index] = pe_dict["queue"][core_index][:queue_index] + [placing_slot] + \
                                           pe_dict["queue"][core_index][queue_index + 1:]

            # updating the live status of the state after the schedule
            # TODO robust
            job_dict["runningTasks"].append(task_ID)
            job_dict["remainingTasks"].remove(task_ID)

            # updating the pre processor queue
            self.preprocessor.queue.remove(task_ID)

        capacitance = pe["capacitance"]
        pe_dict["energyConsumption"][core_index] = capacitance * (volt * volt) * freq
        e = capacitance * (volt * volt) * freq * execution_time
        return reward_function(t=execution_time + lag_time, e=e), fail_flags, e, execution_time + lag_time

    def save_agent_log(self, assigned_job, dict, path_history):
        with self.lock:
            self.agent_log[assigned_job] = dict
            self.paths.append(path_history)

    def assign_job_to_agent(self):
        with self.lock:
            return self.preprocessor.assign_job()

    ####### ENVIRONMENT #######
    def update(self, manager):
        # the state main functionality
        #   1 . getting task window from the window manager if available
        #   2. updating the jobs
        #   3. updating the PEs
        #   4. calling the preprocessor to update the agent queue based on the state
        self.window_manager.run()
        self.__update_jobs(manager)
        self.__update_PEs()
        self.preprocessor.run()

        # displaying the live status of the state
        if self.display:
            print("PEs::")
            pe_data = {}
            for pe_id, pe in self.get_PEs().items():
                pe_data[pe_id] = {
                    "id": pe["id"],
                    "type": pe["type"],
                    "batteryLevel": pe["batteryLevel"],
                    "occupiedCores": list(pe["occupiedCores"]),
                    "energyConsumption": list(pe["energyConsumption"]),
                    "queue": [list(core_queue) for core_queue in pe["queue"]]
                }
            print('\033[94m', pd.DataFrame(pe_data), '\033[0m', '\n')

            print("Jobs::")
            job_data = {}
            for job_id, job in self.get_jobs().items():
                job_data[job_id] = {
                    "task_count": job["task_count"],
                    "finishedTasks": list(job["finishedTasks"]),
                    "runningTasks": list(job["runningTasks"]),
                    "remainingTasks": list(job["remainingTasks"]),
                    "remainingDeadline": job["remainingDeadline"]
                }
            print('\033[92m', pd.DataFrame(job_data), '\033[0m', "\n")

            print(
                "|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")

    ########  UPDATE JOBS #######

    def __update_jobs(self, manager):
        # updating the jobs live status

        # adding new jobs from the window manager to state if there is any
        self.__add_new_active_jobs(self.task_window, manager)

        removing_items = []
        for job in self.jobs.values():
            # decreasing the deadline of the jobs by 1 cycle
            job["remainingDeadline"] -= 1

            # removing the finished jobs from the state
            if len(job["finishedTasks"]) == job["task_count"]:
                removing_items.append(job['id'])

        for item in removing_items:
            del self.jobs[item]

    def __add_new_active_jobs(self, new_tasks, manager):
        if self.display:
            print(f"new window {new_tasks}")
        # updating the jobs in the live status from the task window
        for task_id in new_tasks:
            # Multiprocess Robustness
            task = self.db_tasks[task_id]
            job = self.db_jobs[task['job_id']]

            # initaling job if not; and if appending the new tasks that arrived
            if not self.__is_active_job(task['job_id']):
                self.set_jobs([job], manager)
            self.jobs[task['job_id']]["remainingTasks"].append(task_id)

    def __is_active_job(self, job_ID):
        for job_id in self.jobs.keys():
            if job_ID == job_id:
                return True
        return False

    ####### UPDATE PEs #######
    def __update_PEs(self):
        for pe_index,pe_dict in enumerate(self.PEs):
            # Multiprocess Robustness

            # updating the PEs live status
            #  1. updating the quques

            self.__update_PEs_queue(pe_dict)
            self.__update_occupied_cores(pe_dict,pe_index)

    def __update_occupied_cores(self, pe_dict,pe_index):
        # Multiprocess Robustness
        pe = self.db_devices[pe_index]

        occupied_cores = pe_dict["occupiedCores"]
        energy_consumption = pe_dict["energyConsumption"]
        queue_list = pe_dict["queue"]
        power_idle = pe["powerIdle"]

        for core_index, (occupied, queue) in enumerate(zip(occupied_cores, queue_list)):
            if queue[0] == (0, -1):  # If core is free
                occupied_cores[core_index] = 0
                energy_consumption[core_index] = power_idle
            else:  # Core is occupied
                occupied_cores[core_index] = 1

    def __update_energy_consumption(self, pe, pe_ID):
        for core_index, core_av in enumerate(pe["occupiedCores"]):
            if core_av == 0:
                pe["energyConsumption"][core_index] = self.db_devicespe_ID[
                    "powerIdle"]

    def __update_PEs_queue(self, pe):

        for core_index, core_queue in enumerate(pe["queue"]):
            first_task = core_queue[0]

            if first_task[0] == 0:
                # Removing the finished task from queue
                if first_task[1] != -1:
                    self.__task_finished(first_task[1])
                    # Shift queue and append (0,-1)
                    pe["queue"][core_index] = core_queue[1:] + [(0, -1)]

            else:
                # updating the queue for mec and iot (reducing the remaining time by 1 clock)
                slot = (pe["queue"][core_index][0][0] - 1, pe["queue"][core_index][0][1])
                pe["queue"][core_index] = [slot] + pe["queue"][core_index][1:]

    def __remove_unused_cores_cloud(self, pe, core_list):
        for i, item in enumerate(core_list):
            del pe["queue"][item - i]
            del pe["occupiedCores"][item - i]
            del pe["energyConsumption"][item - i]

    def __task_finished(self, task_ID):
        # Multiprocess Robustness
        try:
            task = self.db_tasks[task_ID]
            job_ID = task["job_id"]
            job = self.jobs[job_ID]
            task_suc = task['successors']
        except:
            return
        if job is None or task_ID not in job["runningTasks"]:
            return

        # updating the predecessors count for the successors tasks of the finished task(ready state)
        for t in task_suc:
            self.db_tasks[t]['pred_count'] -=1
            if self.db_tasks[t]['pred_count'] <= 0:
                # if the task is in the state and the dependencies meet; added it to the queue
                if t in job['remainingTasks']:
                    self.preprocessor.queue.append(t)

        # adding the task to the finished and removing it from the running tasks
        job["finishedTasks"].append(task_ID)
        job["runningTasks"].remove(task_ID)

    def find_place(self, pe, core_i):
        try:
            lag_time = 0
            if pe['type'] == 'cloud' or True:
                for core_index, queue in enumerate(pe["queue"]):
                    if queue[0][1] == -1:
                        
                        return 0, core_index, 0
            return -1, -1, -1
        except:
            print("Retrying find place")
            return self.find_place(pe,core_i)
        # if pe['type'] == 'cloud' and pe["queue"][core_i][0][1] != -1 and core_i < 127:
        #     return self.find_place(pe, core_i + 1)
    
        for i, slot in enumerate(pe["queue"][core_i]):
            if slot[1] == -1:
                lag_time = sum([time for time, taskIndex in pe["queue"][core_i][0:i]])
                return i, core_i, lag_time
        return -1, -1, -1

