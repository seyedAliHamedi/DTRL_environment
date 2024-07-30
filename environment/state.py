import math
from data.configs import summary_log_string, monitor_config
import pandas as pd
import numpy as np
from data.db import Database
from utilities.monitor import Monitor


class State:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._PEs = {}
            cls._instance._jobs = {}
            cls._instance._task_window = {}
            cls._instance.display = False
        return cls._instance

    def initialize(self, display):
        self.display = display
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
        pe = self._PEs[pe_ID]
        pe_database = Database.get_device(pe_ID)
        acceptable_tasks = pe_database["acceptableTasks"]
        task = Database.get_task(task_ID)

        fail_flag = 0
        if (task["task_kind"] not in acceptable_tasks) and (task["is_safe"] and not pe_database['handleSafeTask']):
            fail_flag = 2
            return self.reward_function(punish=True), fail_flag, 0, 0
        elif Database.get_task(task_ID)["is_safe"] and not pe_database['handleSafeTask']:
            fail_flag = 1
            return self.reward_function(punish=True), fail_flag, 0, 0
        elif task["task_kind"] not in acceptable_tasks:
            fail_flag = 1
            return self.reward_function(punish=True), fail_flag, 0, 0

        execution_time = t = math.ceil(Database.get_task(task_ID)[
            "computational_load"] / freq)
        placing_slot = (execution_time, task_ID)
        queue_index, core_index = find_place(pe, core_i)

        # if queue_index == -1:
        #     return self.reward_function(punish=-100)

        # apply on queue
        pe["queue"][core_index][queue_index] = placing_slot
        job_ID = task["job_id"]
        job = self._jobs[job_ID]
        job["assignedTask"] = task_ID

        # remove new assigned task from target job remaining tasks and add it to running tasks
        print(f"trying to remove task{job['assignedTask']} from job{job_ID}| current runningTask{job['runningTasks']}")

        job["remainingTasks"].remove(job["assignedTask"])
        job["runningTasks"].append(task_ID)

        # apply energyConsumption
        if pe['type'] == 'cloud':
            pe["energyConsumption"][core_index] = volt
            e = volt * t
        else:
            capacitance = Database.get_device(pe_ID)["capacitance"][core_index]
            pe["energyConsumption"][core_index] = capacitance * \
                (volt * volt) * freq
            e = capacitance * (volt * volt) * freq * t

        return self.reward_function(e=e, alpha=1, t=t, beta=1), fail_flag, e, t

    def reward_function(self, e=0, alpha=0, t=0, beta=0, punish=0):
        if punish == 0:
            return np.exp(-1 * (e + t))
        else:
            return -100

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

        if self.display:
            print("PEs::")
            print(pd.DataFrame(self._PEs), '\n')
            print("Jobs::")
            print(pd.DataFrame(self._jobs), "\n")
            print(
                "|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")

        if monitor_config['settings']['main']:
            Monitor().add_log('PEs::', start='\n\n', end='')
            Monitor().add_log(f'{pd.DataFrame(self._PEs).to_string()}')
            Monitor().add_log("Jobs::", start='\n', end='')
            Monitor().add_log(f'{pd.DataFrame(self._jobs).to_string()}')
            Monitor().add_log(summary_log_string, start='\n')

    ########  UPDATE JOBS #######
    def __update_jobs(self):

        self.__add_new_active_jobs(self._task_window)

        removing_items = []
        for job_ID in self._jobs.keys():

            # update deadlines
            job = self._jobs[job_ID]
            job["remainingDeadline"] -= 1

            # check for finished jobs
            if len(job["finishedTasks"]) == job["task_count"]:
                removing_items.append(job_ID)

        self.__remove_finished_active_jobs(removing_items)

    def __add_new_active_jobs(self, new_tasks):
        if self.display:
            print(f"new window{new_tasks}")
        for task in new_tasks:
            job_id = Database.get_task(task)["job_id"]
            if not self.__is_active_job(job_id):
                self._set_jobs([Database.get_job(job_id)])
            self.__add_task_to_active_job(task, job_id)

    def __add_task_to_active_job(self, task, job_id):
        self._jobs[job_id]["remainingTasks"].append(task)

    def __is_active_job(self, job_ID):
        for job_ID_key in self._jobs.keys():
            if job_ID_key == job_ID:
                return True
        return False

    def __remove_finished_active_jobs(self, removing_items):
        for item in removing_items:
            del self._jobs[item]

    def __remove_assigned_task(self):
        for job in self._jobs.values():
            job["assignedTask"] = None

    ####### UPDATE PEs #######
    def __update_PEs(self):
        for pe_ID in self._PEs.keys():
            pe = self._PEs[pe_ID]
            self.__update_PEs_queue(pe)
            self.__update_occupied_cores(pe, pe_ID)
            self.__update_batteries_capp(pe)

    def __update_batteries_capp(self, pe):
        if pe["type"] == "mec" or pe["type"] == "cloud":
            return
        pe["batteryLevel"] -= sum(pe["energyConsumption"])

    def __update_energy_consumption(self, pe, pe_ID):
        for core_index, core_av in enumerate(pe["occupiedCores"]):
            if core_av == 0:
                pe["energyConsumption"][core_index] = Database.get_device(pe_ID)["powerIdle"][
                    core_index]

    def __update_PEs_queue(self, pe):
        deleting_queues_on_pe = []
        for core_index, core_queue in enumerate(pe["queue"]):
            current_queue = core_queue
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
                current_queue[0] = (
                    current_queue[0][0] - 1, current_queue[0][1])
        self.__remove_unused_cores_cloud(pe, deleting_queues_on_pe)

    def __remove_unused_cores_cloud(self, pe, core_list):
        for i, item in enumerate(core_list):
            del pe["queue"][item - i]
            del pe["occupiedCores"][item - i]
            del pe["energyConsumption"][item - i]

    def __task_finished(self, task_ID):
        job_ID = Database.get_task(task_ID)["job_id"]
        task_pred = Database.get_task(task_ID)['predecessors']
        for selected_task in task_pred:
            Database.get_task(selected_task)['isReady'] += 1
        try:
            self._jobs[job_ID]["finishedTasks"].append(task_ID)
            self._jobs[job_ID]["runningTasks"].remove(task_ID)
        except:
            print(f"error {task_ID} | job {job_ID}")
            raise

    def __update_occupied_cores(self, pe, pe_ID):
        for core_index, core in enumerate(pe["occupiedCores"]):
            if is_core_free(pe["queue"][core_index]):
                pe["occupiedCores"][core_index] = 0
                # set default energy cons for idle cores
                pe["energyConsumption"][core_index] = Database.get_device(pe_ID)["powerIdle"][
                    core_index]
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
