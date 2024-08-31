import numpy as np
import pandas as pd
from data.db import Database
from environment.pre_processing import Preprocessing
from environment.window_manager import WindowManager
import torch.multiprocessing as mp


class State:

    def __init__(self, display, manager):
        self.database = Database()
        # the state live values PEs & Jobs
        self._PEs = manager.dict()
        self._jobs = manager.dict()
        # the task window manged by the window manager
        self._task_window = manager.list()
        # intilizing PEs in Idle from database
        self._init_PEs(self.database.get_all_devices(), manager)
        # intilizing the preprocessor and the window manager
        self.preprocessor = Preprocessing(state=self, manager=manager)
        self.window_manager = WindowManager(state=self, manager=manager)
        self.agent_log = manager.dict({})
        self.display = display
        self.lock = mp.Lock()

    ###### getters & setters ######
    def get(self):
        return self._jobs, self._PEs

    def get_jobs(self):
        return self.get()[0]

    def get_PEs(self):
        return self.get()[1]

    def set_task_window(self, task_window):
        with self.lock:
            self._task_window = task_window

    def get_task_window(self):
        with self.lock:
            return self._task_window

    def get_job(self, job_id):
        return self.get_jobs().get(job_id)

    ##### Functionality
    def apply_action(self, pe_ID, core_i, freq, volt, task_ID):
        
        # retriving the data from the database and the live values from state
        with self.lock:
            pe_dict = self.get_PEs()[pe_ID]
            pe = self.database.get_device(pe_ID)
            task = self.database.get_task(task_ID)
            job_ID = task["job_id"]
            job_dict = self.get_jobs().get(job_ID)
        
        # calculating the execution time
        execution_time = t = np.ceil(task["computational_load"] / freq)
        
        # placing_slot = (execution_time, task_ID)
        placing_slot = (1, task_ID)
        
        #finding the empty slot in queue for the selected device & core
        queue_index, core_index = find_place(pe_dict, core_i)
        fail_flag = 0
        if (task["is_safe"] and not pe['handleSafeTask']):
            # fail : assigned safe task to unsafe device
            fail_flag += 1
        elif task["task_kind"] not in pe["acceptableTasks"]:
            # fail : assigned a kind of task to the inappropriate device
            fail_flag += 1
        elif queue_index == -1 and core_index == -1:
            # fail : assigned a task to a full queue core
            fail_flag += 1
        # manage failed assingments
        if fail_flag:
            return reward_function(punish=True), fail_flag, 0, 0
        
        # updating the queue slots
        pe_dict["queue"][core_index] = [placing_slot]+  pe_dict["queue"][core_index][1:]
        
        # updating the live status of the state after the schedule
        with self.lock:
            job_dict["runningTasks"].append(task_ID)
            job_dict["remainingTasks"].remove(task_ID)
        
        # updating the pre processor queue
        
        self.preprocessor.queue.remove(task_ID)
        while task_ID in self.preprocessor.queue:
            self.preprocessor.queue.remove(task_ID)
            
        
        # calc and power consumption for different devices
        if pe_dict['type'] == 'cloud':
            pe_dict["energyConsumption"][core_index] = volt
            e = volt * t
        else:
            capacitance = self.database.get_device(
                pe_ID)["capacitance"][core_index]
            pe_dict["energyConsumption"][
                core_index] = capacitance * (volt * volt) * freq
            e = capacitance * (volt * volt) * freq * t
            
        # returning the results
        return reward_function(e=e, t=t), fail_flag, e, t

    def _init_PEs(self, PEs, manager):
        # initlizeing the PEs live variable from db
        for pe in PEs:
            self._PEs[pe["id"]] = {
                "id": pe["id"],
                "type": pe["type"],
                "batteryLevel": pe["battery_capacity"],
                "occupiedCores": manager.list([0 for _ in range(pe["num_cores"])]),
                "energyConsumption": manager.list(pe["powerIdle"]),
                "queue": manager.list([manager.list([(0, -1) for _ in range(pe["maxQueue"])]) for _ in range(pe["num_cores"])]),
            }

    def _set_jobs(self, jobs, manager):
        # add new job the state live status (from window manager)
        for job in jobs:
            self._jobs[job["id"]] = {
                "task_count": job["task_count"],
                "finishedTasks": manager.list([]),
                "runningTasks": manager.list([]),
                "remainingTasks": manager.list([]),
                "remainingDeadline": job["deadline"],
            }

    ####### ENVIRONMENT #######

    def update(self, manager):
        # the state main functionality
        #   1 . getting task window from the window manager if avaliable
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
                    # "type": pe["type"],
                    # "batteryLevel": pe["batteryLevel"],
                    # "occupiedCores": list(pe["occupiedCores"]),
                    # "energyConsumption": list(pe["energyConsumption"]),
                    "queue": [list(core_queue) for core_queue in pe["queue"]]
                }
            print(pd.DataFrame(pe_data).T, '\n')

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
            print(pd.DataFrame(job_data), "\n")

            print(
                "|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")

    ########  UPDATE JOBS #######

    def __update_jobs(self, manager):
        # updating the jobs live status
        
        # adding new jobs from the window manager to state if there is any
        self.__add_new_active_jobs(self._task_window, manager)

        removing_items = []
        for job_ID in self.get_jobs().keys():
            # decreasing the deadline of the jobs by 1 cycle
            job = self.get_jobs().get(job_ID)
            job["remainingDeadline"] -= 1

            # removing the finished jobs from the state
            if len(job["finishedTasks"]) == job["task_count"]:
                removing_items.append(job_ID)

        jobs = self.get_jobs()  
        for item in removing_items:
            del jobs[item]

    def __add_new_active_jobs(self, new_tasks, manager):
        if self.display:
            print(f"new window {new_tasks}")
        # updating the jobs in the live status from the task window
        for task in new_tasks:
            job_id = self.database.get_task(task)["job_id"]
            # initaling job if not; and if appending the new tasks that arrived
            if not self.__is_active_job(job_id):
                self._set_jobs([self.database.get_job(job_id)], manager)
            with self.lock:
                self.get_jobs().get(job_id)["remainingTasks"].append(task)
        
    def __is_active_job(self, job_ID):
        # checking if the job is initlized in the live status or not
        for job_ID_key in self.get_jobs().keys():
            if job_ID_key == job_ID:
                return True
        return False

    ####### UPDATE PEs #######
    def __update_PEs(self):
        for pe_ID in self.get_PEs().keys():
            pe = self.get_PEs()[pe_ID]
            # updating the PEs live status
            #  1. updating the quques
            # TODO occupation ?? queue
            #  2. updating the core occupations
            #  3. updating the battery capcities
            self.__update_PEs_queue(pe)
            self.__update_occupied_cores(pe)
            self.__update_batteries_capp(pe)


    def __update_occupied_cores(self, pe_dict):
        pe = self.database.get_device(pe_dict['id'])
        for core_index, _ in enumerate(pe_dict["occupiedCores"]):
            queue = pe_dict["queue"][core_index]
            # seting the core free and adjusting the power consumption to the Idle mode
            if queue[0] == (0, -1):
                pe_dict["occupiedCores"][core_index] = 0
                pe_dict["energyConsumption"][core_index] = pe["powerIdle"][core_index]
        else:
            # core occupied
            pe_dict["occupiedCores"][core_index] = 1


    def __update_batteries_capp(self, pe):
        if pe["type"] == "iot" :
            pe["batteryLevel"] -= sum(pe["energyConsumption"])

    def __update_energy_consumption(self, pe, pe_ID):
        for core_index, core_av in enumerate(pe["occupiedCores"]):
            if core_av == 0:
                pe["energyConsumption"][core_index] = self.database.get_device(pe_ID)[
                    "powerIdle"][core_index]

    def __update_PEs_queue(self, pe):
        deleting_queues_on_pe = []

        for core_index, _ in enumerate(pe["queue"]):
            if pe["queue"][core_index][0][0] == 0:
                # removing the finished task from queue
                if pe["queue"][core_index][0][1] != -1:
                    self.__task_finished(pe["queue"][core_index][0][1])
                    pe["queue"][core_index] =  pe["queue"][core_index][1:] + [(0,0)]
                # TODO cloud shit
                if pe["type"] == "cloud":
                    deleting_queues_on_pe.append(core_index)
                    continue
            else:
                # updating the queue for mec and iot (reducing the remaining time by 1 clock)
                slot = (pe["queue"][core_index][0][0] -1, pe["queue"][core_index][0][1])
                pe["queue"][core_index] = [
                    slot] + pe["queue"][core_index][1:]

        self.__remove_unused_cores_cloud(pe, deleting_queues_on_pe)

    def __remove_unused_cores_cloud(self, pe, core_list):
        for i, item in enumerate(core_list):
            del pe["queue"][item - i]
            del pe["occupiedCores"][item - i]
            del pe["energyConsumption"][item - i]

    def __task_finished(self, task_ID):
        with self.lock:
            task = self.database.get_task(task_ID)
            job_ID = task["job_id"]
            job = self.get_jobs().get(job_ID)
            task_suc = task['successors']

        # updating the predecessors count for the successors tasks of the finished task(ready state)
        for t in task_suc:
            self.database.task_pred_dec(t)
            if self.database.get_task(t)['pred_count'] == 0:
                # if the task is in the state and the dependencies meet; added it to the queue
                if t in job['remainingTasks']:
                    with self.lock:
                        self.preprocessor.queue.append(t)
        # adding the task to the finisheds and removing it from the runnigs
        
        with self.lock:
            if task_ID==0 and 0 not in job["runningTasks"]:
                return
            job["finishedTasks"].append(task_ID)
            job["runningTasks"].remove(task_ID)

####### UTILITY #######
def reward_function(setup=5, e=0, alpha=1, t=0, beta=1, punish=0):
    if punish:
        return -10

    if setup == 1:
        return -1 * (alpha * e + beta * t)
    elif setup == 2:
        return 1 / (alpha * e + beta * t)
    elif setup == 3:
        return -np.exp(alpha * e) - np.exp(beta * t)
    elif setup == 4:
        return -np.exp(alpha * e + beta * t)
    elif setup == 5:
        return np.exp(-1 * (alpha * e + beta * t))
    elif setup == 6:
        return -np.log(alpha * e + beta * t)
    elif setup == 7:
        return -((alpha * e + beta * t) ** 2)


def find_place(pe, core_i):
    if pe["type"] == "cloud":
        pe["queue"].append([])
        pe["queue"][-1].append((0, -1))
        pe["energyConsumption"].append(0)
        pe["occupiedCores"].append(1)
        return 0, len(pe["queue"]) - 1
    else:
        for i, slot in enumerate(pe["queue"][core_i]):
            if slot == (0, -1):
                return i, core_i
    return -1, -1



def queue_shift_left(queue):
    queue.pop(0)
    queue.append((0, -1))
