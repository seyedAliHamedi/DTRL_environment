import math

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

    def set_task_window(self,task_window):
        self._task_window = task_window

    def apply_action(self, pe_ID, core_index, freq, volt, task_ID):
        last_queue_slot_index =  find_last_queue_slot_index(self._PEs[pe_ID]["queue"][core_index])
        if last_queue_slot_index == -1:
            return False
        # apply on queue
        execution_time = math.ceil(Database.get_task(task_ID)["execution_time"] / freq)
        placing_slot = (execution_time, task_ID)
        self._PEs[pe_ID]["queue"][core_index][last_queue_slot_index] = placing_slot
        job_ID = Database.get_task(task_ID)["job_ID"]
        self._jobs[job_ID]["assignedTask"] = task_ID
        # apply energyConsumption
        capacitance = Database.get_device(pe_ID)[core_index]["capacitance"]
        self._PEs[pe_ID]["energyConsumption"][core_index] = capacitance * (volt * volt) * freq
        return True

    def _init_PEs(self, PEs):
        for pe in PEs:
            self._PEs[pe["id"]] = {
                "id":pe["id"],
                "type": pe["type"],
                "batteryLevel": pe["battery_capacity"],
                "occupiedCores": [0 for core in range(pe["num_cores"])],
                "energyConsumption": pe["powerIdle"],
                # time,task_id
                "queue": [
                    [(0, -1) for _ in range(pe["maxQueue"])]
                    for core in range(pe["num_cores"])
                ],
            }

    def _set_jobs(self, jobs):
        for job in jobs:
            self._jobs[job["id"]] = {
                "task_count": job["task_count"],
                "finishedTasks": [],
                "assignedTask": None,
                "runningTasks": [],
                "remainingTasks": job["tasks_ID"],
                "remainingDeadline": job["deadline"],
            }

 
    ####### ENVIRONMENT #######
    def update(self):

        # process 1
        self.__update_jobs()
        # process 2
        self.__update_PEs()

        self.__remove_assigned_task()

        # print("PEs::")
        # print(self._PEs)
        # print("Jobs::")
        # print(self._jobs, "\n")
        print("|||||||||||||||||||||||||||||||")

    ########  UPDATE JOBS ####### 
    def __update_jobs(self):
        self.__update_deadlines()
        self.__add_new_active_jobs(self._task_window)
        self.__update_remaining_tasks()
        self.__update_running_tasks()
        self.__remove_finished_active_jobs()

    def __add_new_active_jobs(self, new_tasks):
        for task in new_tasks:
            job_id = Database.get_task(task)["job_id"]
            if not self.__is_active_job(job_id):
                self._set_jobs([Database.get_job(job_id)])

    def __is_active_job(self, job_ID):
        for job_ID_key in self._jobs.keys():
            if job_ID_key == job_ID:
                return True
        return False

    def __update_remaining_tasks(self):
        for job in self._jobs.values():
            if job["assignedTask"]:
                for task in job["remainingTasks"]:
                    if task == job["assignedTask"]:
                        job["remainingTasks"].remove(task)

    def __update_running_tasks(self):
        for job_ID in self._jobs.keys():
            if self._jobs[job_ID]["assignedTask"]:
                self._jobs[job_ID]["runningTasks"].append(self._jobs[job_ID]["assignedTask"])

    def __remove_finished_active_jobs(self):
        for job_ID in self._jobs.keys():
            if len(self._jobs[job_ID]["remainingTasks"]) == 0:
                del self._jobs[job_ID]

                

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
        # time
        for pe in self._PEs.values():
            pe["batteryLevel"] -= sum(pe["energyConsumption"])

    def __update_energy_consumption(self):
        for pe_ID in self._PEs.keys():
            for core_index, core_av in enumerate(self._PEs[pe_ID]["occupiedCores"]):
                if core_av == 0:
                    self._PEs[pe_ID]["energyConsumption"][core_index] = Database.get_device(pe_ID)["powerIdle"][core_index]

    def __update_PEs_queue(self):
        for pe in self._PEs.values():
            for core_index, core_queue in enumerate(pe["queue"]):
                current_queue = pe["queue"][core_index]
                # if time of this slot in queue is 0
                if core_queue[0][0] == 0:
                     queue_shift_left(current_queue)
                else:
                    current_queue[0][0] -= 1

    def __update_occupied_cores(self):
        # based on pe queue
        for pe in self._PEs.values():
            for core_index, core in enumerate(pe["occupiedCores"]):
                if  is_core_free(pe["queue"][core_index]):
                    pe["occupiedCores"][core_index] = 0
                else:
                    pe["occupiedCores"][core_index] = 1


####### UTILITY #######     
def find_last_queue_slot_index(queue):
    for i, slot in enumerate(queue):
        if slot == (0, -1):
            return i
    return -1


def is_core_free(queue):
    if queue[0] == (0, -1):
        return True
    else:
        return False


def queue_shift_left(queue):
    queue.pop(0)
    queue.append((0, -1))
