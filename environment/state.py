import math

import pandas as pd

from data.db import Database


class State:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._PEs = {}
            cls._instance._jobs = {}
            cls._instance._task_window = {}
        return cls._instance

    def initialize(self):
        self._init_PEs(Database.get_all_devices())

    def get(self):
        return self._jobs, self._PEs

    def set_task_window(self, task_window):
        self._task_window = task_window

    def get_task_window(self):
        return self._task_window

    def get_job(self, job_id):
        return self._jobs[job_id]

    def apply_action(self, pe_ID, core_i, freq, volt, task_ID):
        execution_time = math.ceil(Database.get_task(task_ID)["computational_load"] / freq)
        placing_slot = (execution_time, task_ID)
        queue_index, core_index = find_place(self._PEs[pe_ID], core_i)

        if queue_index == -1:
            # ! punishment exceed queue
            return False

        # apply on queue
        self._PEs[pe_ID]["queue"][core_index][queue_index] = placing_slot
        job_ID = Database.get_task(task_ID)["job_id"]
        self._jobs[job_ID]["assignedTask"] = task_ID

        # apply energyConsumption
        if self._PEs[pe_ID]['type'] == 'cloud':
            self._PEs[pe_ID]["energyConsumption"][core_index] = volt
        else:
            capacitance = Database.get_device(pe_ID)["capacitance"][core_index]
            self._PEs[pe_ID]["energyConsumption"][core_index] = capacitance * (volt * volt) * freq

        # ! reward: e+t
        return True

    def _init_PEs(self, PEs):
        for pe in PEs:
            self._PEs[pe["id"]] = {
                "id": pe["id"],
                "type": pe["type"],
                "batteryLevel": pe["battery_capacity"],
                "occupiedCores": [0 for core in range(pe["num_cores"])],
                "energyConsumption": pe["powerIdle"],
                # queue:
                # core 1 queue ...
                # core 2 queue ...
                # each queue element: (execution_time,task_id)
                "queue": [[(0, -1) for _ in range(pe["maxQueue"])] for core in range(pe["num_cores"])],
            }

    def _set_jobs(self, jobs):
        for job in jobs:
            self._jobs[job["id"]] = {
                "task_count": job["task_count"],
                "finishedTasks": [],
                "assignedTask": None,
                "runningTasks": [],
                "remainingTasks": [],
                "remainingDeadline": job["deadline"],
            }

    ####### ENVIRONMENT #######
    def update(self):

        # process 1
        self.__update_jobs()
        # process 2
        self.__update_PEs()

        self.__remove_assigned_task()

        print("PEs::")
        print(pd.DataFrame(self._PEs), '\n')
        print("Jobs::")
        print(pd.DataFrame(self._jobs), "\n")
        print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")

    ########  UPDATE JOBS ####### 
    def __update_jobs(self):
        self.__update_deadlines()
        self.__add_new_active_jobs(self._task_window)
        self.__update_remaining_tasks()
        self.__update_running_tasks()
        self.__remove_finished_active_jobs()

    def __add_new_active_jobs(self, new_tasks):
        print(f"new window{new_tasks}")
        for task in new_tasks:
            job_id = Database.get_task(task)["job_id"]
            if not self.__is_active_job(job_id):
                self._set_jobs([Database.get_job(job_id)])
            self.__add_task_to_active_job(task)

    def __add_task_to_active_job(self, task):
        job_id = Database.get_task(task)["job_id"]
        self._jobs[job_id]["remainingTasks"].append(task)

    def __is_active_job(self, job_ID):
        for job_ID_key in self._jobs.keys():
            if job_ID_key == job_ID:
                return True
        return False

    def __update_remaining_tasks(self):
        for job in self._jobs.values():
            if job["assignedTask"] or job["assignedTask"] == 0:
                for task in job["remainingTasks"]:
                    if task == job["assignedTask"]:
                        job["remainingTasks"].remove(task)

    def __update_running_tasks(self):
        for job_ID in self._jobs.keys():
            if self._jobs[job_ID]["assignedTask"] or self._jobs[job_ID]["assignedTask"] == 0:
                self._jobs[job_ID]["runningTasks"].append(self._jobs[job_ID]["assignedTask"])

    def __remove_finished_active_jobs(self):
        removing_items = []
        for job_ID in self._jobs.keys():
            selected_job = self._jobs[job_ID]
            if len(selected_job["finishedTasks"]) == selected_job["task_count"]:
                removing_items.append(job_ID)
        for item in removing_items:
            del self._jobs[item]

    def __update_deadlines(self):
        # TODO : if < 0 return punishment ????
        for job in self._jobs.values():
            job["remainingDeadline"] -= 1

    def __remove_assigned_task(self):
        for job in self._jobs.values():
            job["assignedTask"] = None

    ####### UPDATE PEs ####### 
    def __update_PEs(self):
        self.__update_PEs_queue()
        self.__update_occupied_cores()
        self.__update_energy_consumption()
        self.__update_batteries_capp()

    def __update_batteries_capp(self):
        for pe in self._PEs.values():
            if pe["type"] == "mec" or pe["type"] == "cloud":
                continue
            pe["batteryLevel"] -= sum(pe["energyConsumption"])

    def __update_energy_consumption(self):
        for pe_ID in self._PEs.keys():
            for core_index, core_av in enumerate(self._PEs[pe_ID]["occupiedCores"]):
                if core_av == 0:
                    self._PEs[pe_ID]["energyConsumption"][core_index] = Database.get_device(pe_ID)["powerIdle"][
                        core_index]

    def __update_PEs_queue(self):
        for pe in self._PEs.values():
            deleting_queues_on_pe = []
            for core_index, core_queue in enumerate(pe["queue"]):
                current_queue = pe["queue"][core_index]
                # if time of this slot in queue is 0
                if current_queue[0][0] == 0:
                    if current_queue[0][1] != -1:
                        finished_task_ID = current_queue[0][1]
                        self.__task_finished(finished_task_ID)
                    if pe["type"] == "cloud":
                        deleting_queues_on_pe.append(core_index)
                        continue
                    queue_shift_left(current_queue)
                else:
                    current_queue[0] = (current_queue[0][0] - 1, current_queue[0][1])
            self.__remove_unused_cores_cloud(pe, deleting_queues_on_pe)

    def __remove_unused_cores_cloud(self, pe, core_list):
        for i, item in enumerate(core_list):
            del pe["queue"][item - i]
            del pe["occupiedCores"][item - i]
            del pe["energyConsumption"][item - i]

    def __task_finished(self, task_ID):
        job_ID = Database.get_task(task_ID)["job_id"]
        self._jobs[job_ID]["finishedTasks"].append(task_ID)
        self._jobs[job_ID]["runningTasks"].remove(task_ID)

    def __update_occupied_cores(self):
        # based on pe queue
        for pe in self._PEs.values():
            for core_index, core in enumerate(pe["occupiedCores"]):
                if is_core_free(pe["queue"][core_index]):
                    pe["occupiedCores"][core_index] = 0
                else:
                    pe["occupiedCores"][core_index] = 1


####### UTILITY #######     
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


def is_core_free(queue):
    if queue[0] == (0, -1):
        return True
    else:
        return False


def queue_shift_left(queue):
    queue.pop(0)
    queue.append((0, -1))


def is_already_added_to_job(task, job_task_list):
    for item in job_task_list:
        if task == item:
            return True
        else:
            return False
