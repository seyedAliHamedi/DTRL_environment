import math
from data.configs import summary_log_string, monitor_config
import pandas as pd
import numpy as np
from utilities.monitor import Monitor


class State:
    _instance = None

    def __new__(cls, db=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._PEs = {}
            cls._instance._jobs = {}
            cls._instance._task_window = {}
            cls._instance.display = False
            cls._instance.db = db
            cls._instance.done_jobs = 0
        return cls._instance

    def initialize(self, display):
        self.display = display
        self._init_PEs(self.db.get_all_devices())

    def get(self):
        return self._jobs, self._PEs

    def set_task_window(self, task_window):
        self._task_window = task_window

    def get_task_window(self):
        return self._task_window

    def get_job(self, job_id):
        return self._jobs[job_id]

    def get_jobs_len(self):
        return len(self._jobs.keys())

    def apply_action(self, pe_ID, core_i, freq, volt, task_ID):
        pe = self._PEs[pe_ID]
        pe_database = self.db.get_device(pe_ID)
        task = self.db.get_task(task_ID)

        execution_time = np.ceil(task["computational_load"] / freq)

        if execution_time > 5:
            execution_time = 5
        # TODO t must include time of tasks scheduled before it ,in selected queue
        placing_slot = (execution_time, task_ID)

        queue_index, core_index, lag_time = find_place(pe, core_i)

        fail_flags = [0, 0, 0, 0]
        if task["is_safe"] and not pe_database['handleSafeTask']:
            # fail : assigned safe task to unsafe device
            fail_flags[0] = 1
        if task["task_kind"] not in pe_database["acceptableTasks"]:
            # fail : assigned a kind of task to the inappropriate device
            fail_flags[1] = 1
        if queue_index == -1 and core_index == -1:
            # fail : assigned a task to a full queue core
            fail_flags[2] = 1
            # return sum(fail_flags) * reward_function(0, 0, punish=True), fail_flags, 0, 0

        # apply on queue
        pe["queue"][core_index][queue_index] = placing_slot
        job_ID = task["job_id"]
        job = self._jobs[job_ID]
        job["assignedTask"] = task_ID

        # remove new assigned task from target job remaining tasks and add it to running tasks
        job["remainingTasks"].remove(job["assignedTask"])
        job["runningTasks"].append(task_ID)

        # apply energyConsumption
        capacitance = pe_database["capacitance"][core_index]
        pe["energyConsumption"][core_index] = capacitance * (volt * volt) * freq
        e = capacitance * (volt * volt) * freq * execution_time

        if sum(fail_flags) > 0:
            return sum(fail_flags) * reward_function(0, 0, punish=True), fail_flags, 10, 10

        return reward_function(e, execution_time), fail_flags, e, execution_time

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
            print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")

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
                self.done_jobs += 1
                removing_items.append(job_ID)

        self.__remove_finished_active_jobs(removing_items)

    def __add_new_active_jobs(self, new_tasks):
        if self.display:
            print(f"new window{new_tasks}")
        for task in new_tasks:
            job_id = self.db.get_task(task)["job_id"]
            if not self.__is_active_job(job_id):
                self._set_jobs([self.db.get_job(job_id)])
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
                pe["energyConsumption"][core_index] = self.db.get_device(pe_ID)["powerIdle"][
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
                queue_shift_left(current_queue)
            else:
                current_queue[0] = (current_queue[0][0] - 1, current_queue[0][1])

    def __task_finished(self, task_ID):
        job_ID = self.db.get_task(task_ID)["job_id"]
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
                if pe['type'] == 'cloud':
                    pe["energyConsumption"][core_index] = 0
                else:
                    pe["energyConsumption"][core_index] = self.db.get_device(pe_ID)["powerIdle"][
                        core_index]
            else:
                pe["occupiedCores"][core_index] = 1


####### UTILITY #######

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


def find_place(pe, core_i):
    if pe['type'] == 'cloud' or True:
        for core_index, queue in enumerate(pe["queue"]):
            if queue[0][1] == -1:
                return 0, core_index, 0
    # for i, slot in enumerate(pe["queue"][core_i]):
    #     if slot[1] == -1:
    #         lag_time = sum([time for time, taskIndex in pe["queue"][core_i][0:i]])
    #         return i, core_i, lag_time
    return -1, -1, -1


def reward_function(e, t, alpha=1, beta=1, punish=False):
    if punish is True:
        return -10
    else:
        return np.exp(-1 * (e * alpha + t * beta)) + 1
