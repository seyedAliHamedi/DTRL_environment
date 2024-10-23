from collections import deque
import numpy as np
import pandas as pd
import torch.multiprocessing as mp



from environment.util import *
from environment.pre_processing import Preprocessing
from environment.window_manager import WindowManager

from configs import environment_config


class State:
    def __init__(self, devices, jobs, tasks, manager):
        self.db_devices = devices
        self.db_jobs = jobs
        self.db_tasks = {item['id']: item for item in tasks}
        
        # !! # Use manager.list only for mutable shared data
        self.PEs = manager.list()
        self.jobs = manager.dict()
        self.util = Utility(devices=self.db_devices)
        self.init_PEs(self.db_devices, manager)
        
        self.task_window = manager.list()
        
        self.preprocessor = Preprocessing(state=self, manager=manager)
        self.window_manager = WindowManager(state=self, manager=manager)

        self.agent_log = manager.dict()
        self.paths = manager.list()
        self.display = environment_config['display']
        self.lock = mp.Lock()
        # self.device_usuages = manager.list([manager.list([1]) for i in range(len(self.db_devices))])
        

    ##### Intializations
    def init_PEs(self, PEs, manager):
        # initializing the PEs live variable from db
        for pe in PEs:
            self.PEs.append({
                "type": pe["type"],
                "batteryLevel": 100.0,
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
    
    def remove_job(self,job_id):
        try:
            with self.lock:
                del self.jobs[job_id]
        except:
            return
    
    def clean_jobs(self):
        return
        removing_items = []
        for job in self.jobs.values():
            if job['remainingDeadline'] < 0 and job['id'] not in removing_items:
                print("removed deadline")
                removing_items.append(job['id'])
                
        for item in removing_items:
            if item in self.jobs.keys():
                del self.jobs[item]
        
    ##### Functionality
    def apply_action(self, pe_ID, core_i, freq, volt, task_ID, utilization=None, diversity=None, gin=None):
        try:
            pe_dict = self.PEs[pe_ID]
            pe = self.db_devices[pe_ID]
            task = self.db_tasks[task_ID]
            if task["job_id"] not in self.jobs.keys():
                return 0, [0,0,0,0], 0, 0
            job_dict = self.jobs[task["job_id"]]
        except:
            print("Retry apply action")
            return self.apply_action(pe_ID, core_i, freq, volt, task_ID, utilization, diversity, gin)
        with self.lock:
            total_t, total_e  = calc_total(pe,task,[self.db_tasks[pre_id] for pre_id in task["predecessors"]],core_i,0)

            if total_t > 1:
                total_t = 1

            placing_slot = (total_t, task_ID)

            queue_index, core_index, lag_time = self.find_place(pe_dict, core_i)

            fail_flags = [0, 0, 0, 0]
            if task["is_safe"] and not pe['is_safe']:
                fail_flags[0] = 1
            if task["task_kind"] not in pe["acceptable_tasks"]:
                fail_flags[1] = 1
            if queue_index == -1 and core_index == -1:
                fail_flags[2] = 1

            if sum(fail_flags) > 0:
                return sum(fail_flags) * reward_function(punish=True), fail_flags, 0, 0


            # for i, _ in enumerate(self.device_usuages):
            #     if i == pe_ID:
            #         self.device_usuages[i].append(1)
            #     else:
            #         self.device_usuages[i].append(0)
            #     if len(self.device_usuages[i])>100:
            #         self.device_usuages[i][:]=self.device_usuages[i][1:]
                
            
            pe_dict["queue"][core_index] = pe_dict["queue"][core_index][:queue_index] + [placing_slot] + \
                                           pe_dict["queue"][core_index][queue_index + 1:]

            job_dict["runningTasks"].append(task_ID)
            try:
                job_dict["remainingTasks"].remove(task_ID)
            except:
                pass

            self.preprocessor.queue.remove(task_ID)

        
        battery_punish, batteryFail=self.util.checkBatteryDrain(total_e,pe_dict,pe) 
        if batteryFail:
            fail_flags[3]=1
            return sum(fail_flags) * reward_function(punish=True), fail_flags, 0, 0
        
        lambda_penalty = 0
        if learning_config['utilization']            :
            lambda_diversity = learning_config["max_lambda"] * (1 - diversity)
            lambda_gini = learning_config["max_lambda"] * gin
            lambda_penalty = learning_config["alpha_diversity"] * lambda_diversity + learning_config["alpha_gin"] * lambda_gini

        return reward_function(t=total_t + lag_time, e=total_e) +battery_punish, fail_flags, total_e, total_t+ lag_time


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
        # self.clean_jobs()
        self.__update_PEs()
        self.preprocessor.run()

        # displaying the live status of the state
        if self.display :
            print("PEs::")
            pe_data = {}
            for pe_id, pe in enumerate(self.PEs):
                pe_data[pe_id] = {
                    "id": pe_id,
                    # "type": pe["type"],
                    # "batteryLevel": pe["batteryLevel"],
                    # "occupiedCores": list(pe["occupiedCores"]),
                    # "energyConsumption": list(pe["energyConsumption"]),
                    "queue": [list(core_queue) for core_queue in pe["queue"]]
                }
            print('\033[94m', pd.DataFrame(pe_data), '\033[0m', '\n')

            print("Jobs::")
            job_data = {}
            for job_id, job in self.jobs.items():
                job_data[job_id] = {
                    # "task_count": job["task_count"],
                    # "finishedTasks": list(job["finishedTasks"]),
                    "runningTasks": list(job["runningTasks"]),
                    # "remainingTasks": list(job["remainingTasks"]),
                    # "remainingDeadline": job["remainingDeadline"]
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
        for job_id in list(self.jobs.keys()):
            job = self.jobs[job_id]
            # decreasing the deadline of the jobs by 1 cycle
            old_deadline = job["remainingDeadline"]
            updated_job = dict(job)
            updated_job["remainingDeadline"] = old_deadline - 1
            
            # Replace the old dictionary in the manager.dict()
            self.jobs[job_id] = updated_job

            # removing the finished jobs from the state
            if  len(job["finishedTasks"]) == job["task_count"]:
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

            if first_task[0] <= 0:
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

