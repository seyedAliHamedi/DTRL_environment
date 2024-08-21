import numpy as np
import pandas as pd
from data.db import Database
from environment.pre_processing import Preprocessing
from environment.window_manager import WindowManager
import torch.multiprocessing as mp


class State:

    def __init__(self, display, manager):
        self.database = Database()
        self._PEs = manager.dict()
        self._jobs = manager.dict()
        self._task_window = manager.list()
        self._init_PEs(self.database.get_all_devices(), manager)
        self.preprocessor = Preprocessing(state=self, manager=manager)
        self.window_manager = WindowManager(state=self, manager=manager)
        self.agent_log = manager.dict({})
        self.display = display
        self.lock = mp.Lock()

    def get(self):
        return self._jobs, self._PEs

    def get_jobs(self):
        return self.get()[0]

    def get_PEs(self):
        return self.get()[1]

    def set_task_window(self, task_window):
        self._task_window = task_window

    def get_task_window(self):
        return self._task_window

    def get_job(self, job_id):
        return self.get_jobs().get(job_id)

    def apply_action(self, pe_ID, core_i, freq, volt, task_ID):
        pe_dict = self.get_PEs()[pe_ID]
        pe = self.database.get_device(pe_ID)
        task = self.database.get_task(task_ID)
        execution_time = t = np.ceil(task["computational_load"] / freq)
        # placing_slot = (execution_time, task_ID)
        placing_slot = (1, task_ID)
        queue_index, core_index = find_place(pe_dict, core_i)

        fail_flag = 0
        if (task["is_safe"] and not pe['handleSafeTask']):
            fail_flag += 1
        elif task["task_kind"] not in pe["acceptableTasks"]:
            fail_flag += 1
        elif queue_index == -1 and core_index == -1:
            fail_flag += 1
        if fail_flag:
            return reward_function(punish=True), fail_flag, 0, 0
        # Convert manager.list to regular list for modification
        pe_dict["queue"][core_index] = [placing_slot] + \
            pe_dict["queue"][core_index][1:]
        job_ID = task["job_id"]
        job_dict = self.get_jobs().get(job_ID)
        job_dict["assignedTask"] = task_ID
        try:
            job_dict["remainingTasks"].remove(task_ID)
        except:
            pass
        job_dict["runningTasks"].append(task_ID)
        self.preprocessor.remove_from_queue(task_ID)
        if pe_dict['type'] == 'cloud':
            pe_dict["energyConsumption"][core_index] = volt
            e = volt * t
        else:
            capacitance = self.database.get_device(
                pe_ID)["capacitance"][core_index]
            pe_dict["energyConsumption"][
                core_index] = capacitance * (volt * volt) * freq
            e = capacitance * (volt * volt) * freq * t

        return reward_function(e=e, t=t), fail_flag, e, t

    def _init_PEs(self, PEs, manager):
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
        for job in jobs:
            self._jobs[job["id"]] = {
                "task_count": job["task_count"],
                "finishedTasks": manager.list([]),
                "assignedTask": None,
                "runningTasks": manager.list([]),
                "remainingTasks": manager.list([]),
                "remainingDeadline": job["deadline"],
            }

    ####### ENVIRONMENT #######

    def update(self, manager):
        self.window_manager.run()
        self.__update_jobs(manager)
        self.__update_PEs()
        self.__remove_assigned_task()
        self.preprocessor.run()

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
            print(pd.DataFrame(pe_data).T, '\n')

            print("Jobs::")
            job_data = {}
            for job_id, job in self.get_jobs().items():
                job_data[job_id] = {
                    "task_count": job["task_count"],
                    "finishedTasks": list(job["finishedTasks"]),
                    "assignedTask": job["assignedTask"],
                    "runningTasks": list(job["runningTasks"]),
                    "remainingTasks": list(job["remainingTasks"]),
                    "remainingDeadline": job["remainingDeadline"]
                }
            print(pd.DataFrame(job_data), "\n")

            print(
                "|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")

    ########  UPDATE JOBS #######

    def __update_jobs(self, manager):
        self.__add_new_active_jobs(self._task_window, manager)

        removing_items = []
        for job_ID in self.get_jobs().keys():
            job = self.get_jobs().get(job_ID)
            job["remainingDeadline"] -= 1

            if len(job["finishedTasks"]) == job["task_count"]:
                removing_items.append(job_ID)

        self.__remove_finished_active_jobs(removing_items)

    def __add_new_active_jobs(self, new_tasks, manager):
        if self.display:
            print(f"new window {new_tasks}")
        for task in new_tasks:
            job_id = self.database.get_task(task)["job_id"]
            if not self.__is_active_job(job_id):
                self._set_jobs([self.database.get_job(job_id)], manager)
            self.get_jobs().get(job_id)["remainingTasks"].append(task)

        

    def __is_active_job(self, job_ID):
        for job_ID_key in self.get_jobs().keys():
            if job_ID_key == job_ID:
                return True
        return False

    def __remove_finished_active_jobs(self, removing_items):
        for item in removing_items:
            del self.get_jobs().get(item)

    def __remove_assigned_task(self):
        for job in self.get_jobs().values():
            job["assignedTask"] = None

    ####### UPDATE PEs #######
    def __update_PEs(self):
        for pe_ID in self.get_PEs().keys():
            pe = self.get_PEs()[pe_ID]
            self.__update_PEs_queue(pe)
            self.__update_occupied_cores(pe)
            self.__update_batteries_capp(pe)

    def __update_batteries_capp(self, pe):
        if pe["type"] == "mec" or pe["type"] == "cloud":
            return
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
                if pe["queue"][core_index][0][1] != -1:
                    self.__task_finished(pe["queue"][core_index][0][1])
                if pe["type"] == "cloud":
                    deleting_queues_on_pe.append(core_index)
                    continue
                queue_shift_left(pe["queue"][core_index])
            else:
                slot = (pe["queue"][core_index][0][0] -
                        1, pe["queue"][core_index][0][1])
                pe["queue"][core_index] = [
                    slot] + pe["queue"][core_index][1:]

        self.__remove_unused_cores_cloud(pe, deleting_queues_on_pe)

    def __remove_unused_cores_cloud(self, pe, core_list):
        for i, item in enumerate(core_list):
            del pe["queue"][item - i]
            del pe["occupiedCores"][item - i]
            del pe["energyConsumption"][item - i]

    def __task_finished(self, task_ID):
        task = self.database.get_task(task_ID)
        job_ID = task["job_id"]
        job = None
        job = self.get_jobs().get(job_ID)
        task_suc = task['successors']

        for t in task_suc:
            self.database.task_pred_dec(t)
            if self.database.get_task(t)['pred_count'] == 0:
                if t in job['remainingTasks']:
                    self.preprocessor.queue.append(t)
        job["finishedTasks"].append(task_ID)
        if task_ID in job["runningTasks"]:
            job["runningTasks"].remove(task_ID)

    def __update_occupied_cores(self, pe_dict):
        pe = self.database.get_device(pe_dict['id'])
        for core_index, _ in enumerate(pe_dict["occupiedCores"]):
            if is_core_free(pe_dict["queue"][core_index]):
                pe_dict["occupiedCores"][core_index] = 0
                pe_dict["energyConsumption"][core_index] = pe["powerIdle"][core_index]
        else:
            pe_dict["occupiedCores"][core_index] = 1


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


def is_core_free(queue):
    if queue[0] == (0, -1):
        return True
    else:
        return False


def queue_shift_left(queue):
    queue.pop(0)
    queue.append((0, -1))
