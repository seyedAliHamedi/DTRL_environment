import math
import numpy as np
import pandas as pd
from multiprocessing import Manager, Lock
from data.configs import summary_log_string, monitor_config
from data.db import Database
from environment.pre_processing import Preprocessing
from environment.window_manager import WindowManager
from utilities.monitor import Monitor


class State:
    jobs_done = 0

    def __init__(self, display, manager, lock):
        self.database = Database()
        self._PEs = manager.dict()
        self._jobs = manager.dict()
        self._task_window = manager.list()
        self.initialize(manager)
        self.preprocessor = Preprocessing(state=self, manager=manager)
        self.window_manager = WindowManager(state=self, manager=manager)
        self.lock = lock
        self.display = display

    def initialize(self, manager):
        self._init_PEs(self.database.get_all_devices(), manager)
        print("State Initialization complete.")

    def get(self):
        return self._jobs, self._PEs

    def set_task_window(self, task_window):
        self._task_window = task_window

    def get_task_window(self):
        return self._task_window

    def get_job(self, job_id):
        with self.lock:
            return self._jobs[job_id]

    def apply_action(self, pe_ID, core_i, freq, volt, task_ID):
        with self.lock:
            pe = self._PEs[pe_ID]
            pe_database = self.database.get_device(pe_ID)
            acceptable_tasks = pe_database["acceptableTasks"]
            task = self.database.get_task(task_ID)

            fail_flag = 0
            if (task["task_kind"] not in acceptable_tasks) and (task["is_safe"] and not pe_database['handleSafeTask']):
                fail_flag = 2
                return self.reward_function(punish=True), fail_flag, 0, 0
            elif task["task_kind"] not in acceptable_tasks:
                fail_flag = 1
                return self.reward_function(punish=True), fail_flag, 0, 0
            elif task["task_kind"] not in acceptable_tasks:
                fail_flag = 1
                return self.reward_function(punish=True), fail_flag, 0, 0

            execution_time = t = math.ceil(self.database.get_task(task_ID)[
                                           "computational_load"] / freq)
            placing_slot = (execution_time, task_ID)
            queue_index, core_index = find_place(pe, core_i)

            # if queue_index != -1 and core_index != -1:

            # Convert manager.list to regular list for modification
            queue = list(pe["queue"][core_index])
            queue[queue_index] = placing_slot
            pe["queue"][core_index] = queue
            job_ID = task["job_id"]
            job = self._jobs[job_ID]
            job["assignedTask"] = task_ID

            try:
                if task_ID in job["remainingTasks"]:
                    job["remainingTasks"].remove(task_ID)
                else:
                    print(
                        f"Task ID {task_ID} not found in remaining tasks of job {job_ID}")
            except Exception as e:
                print(f"Exception occurred: {e}")
                raise

            job["runningTasks"].append(task_ID)

            self.preprocessor.remove_from_queue(task_ID)

            if pe['type'] == 'cloud':
                list(pe["energyConsumption"])[core_index] = volt
                e = volt * t
            else:
                capacitance = self.database.get_device(
                    pe_ID)["capacitance"][core_index]
                list(pe["energyConsumption"])[core_index] = capacitance * \
                    (volt * volt) * freq
                e = capacitance * (volt * volt) * freq * t

            return self.reward_function(e=e, alpha=1, t=t, beta=1), fail_flag, e, t

    def reward_function(self, e=0, alpha=0, t=0, beta=0, punish=0):
        if punish == 0:
            return np.exp(-1 * (e + t))
        else:
            return -100

    def _init_PEs(self, PEs, manager):
        for pe in PEs:
            self._PEs[pe["id"]] = {
                "id": pe["id"],
                "type": pe["type"],
                "batteryLevel": pe["battery_capacity"],
                # Use manager.list()
                "occupiedCores": manager.list([0 for _ in range(pe["num_cores"])]),
                # Use manager.list()
                "energyConsumption": manager.list(pe["powerIdle"]),
                # Use manager.list()
                "queue": manager.list([[(0, -1) for _ in range(pe["maxQueue"])] for _ in range(pe["num_cores"])]),
            }

    def _set_jobs(self, jobs, manager):
        for job in jobs:
            self._jobs[job["id"]] = {
                "task_count": job["task_count"],
                "finishedTasks": manager.list([]),  # Use manager.list()
                "assignedTask": None,
                "runningTasks": manager.list([]),  # Use manager.list()
                "remainingTasks": manager.list([]),  # Use manager.list()
                "remainingDeadline": job["deadline"],
            }

    ####### ENVIRONMENT #######

    def update(self, manager):
        with self.lock:
            self.window_manager.run()
            self.__update_jobs(manager)
            self.__update_PEs()
            self.__remove_assigned_task()
            self.preprocessor.run()

            if self.display:
                print("PEs::")
                # Convert manager.dict() to a regular dict for easier printing
                pe_data = {}
                for pe_id, pe in self._PEs.items():
                    pe_data[pe_id] = {
                        "id": pe["id"],
                        "type": pe["type"],
                        "batteryLevel": pe["batteryLevel"],
                        # Convert ListProxy to list
                        "occupiedCores": list(pe["occupiedCores"]),
                        # Convert ListProxy to list
                        "energyConsumption": list(pe["energyConsumption"]),
                        # Convert each ListProxy in queue to list
                        "queue": [list(core_queue) for core_queue in pe["queue"]]
                    }
                # Transpose for better readability
                print(pd.DataFrame(pe_data).T, '\n')

                print("Jobs::")
                # Convert manager.dict() to a regular dict for easier printing
                job_data = {}
                for job_id, job in self._jobs.items():
                    job_data[job_id] = {
                        "task_count": job["task_count"],
                        # Convert ListProxy to list
                        "finishedTasks": list(job["finishedTasks"]),
                        "assignedTask": job["assignedTask"],
                        # Convert ListProxy to list
                        "runningTasks": list(job["runningTasks"]),
                        # Convert ListProxy to list
                        "remainingTasks": list(job["remainingTasks"]),
                        "remainingDeadline": job["remainingDeadline"]
                    }
                print(pd.DataFrame(job_data), "\n")

                print(
                    "|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")

            if monitor_config['settings']['main']:
                Monitor().add_log('PEs::', start='\n\n', end='')
                Monitor().add_log(f'{pd.DataFrame(pe_data).to_string()}')
                Monitor().add_log("Jobs::", start='\n', end='')
                Monitor().add_log(f'{pd.DataFrame(job_data).to_string()}')
                Monitor().add_log(summary_log_string, start='\n')

    ########  UPDATE JOBS #######

    def __update_jobs(self, manager):
        self.__add_new_active_jobs(self._task_window, manager)

        removing_items = []
        for job_ID in self._jobs.keys():
            job = self._jobs[job_ID]
            job["remainingDeadline"] -= 1

            if len(list(job["finishedTasks"])) == job["task_count"]:
                removing_items.append(job_ID)

        self.__remove_finished_active_jobs(removing_items)

    def __add_new_active_jobs(self, new_tasks, manager):
        if self.display:
            print(f"new window {new_tasks}")
        for task in new_tasks:
            job_id = self.database.get_task(task)["job_id"]
            if not self.__is_active_job(job_id):
                self._set_jobs([self.database.get_job(job_id)], manager)
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
        pe["batteryLevel"] -= sum(list(pe["energyConsumption"]))

    def __update_energy_consumption(self, pe, pe_ID):
        for core_index, core_av in enumerate(list(pe["occupiedCores"])):
            if core_av == 0:
                list(pe["energyConsumption"])[core_index] = self.database.get_device(pe_ID)[
                    "powerIdle"][core_index]

    def __update_PEs_queue(self, pe):
        deleting_queues_on_pe = []

        queue_copy = [list(core_queue) for core_queue in pe["queue"]]

        for core_index, core_queue in enumerate(queue_copy):
            current_queue = core_queue
            if current_queue[0][0] == 0:
                if current_queue[0][1] != -1:
                    finished_task_ID = current_queue[0][1]
                    self.__task_finished(finished_task_ID)
                if pe["type"] == "cloud":
                    deleting_queues_on_pe.append(core_index)
                    continue
                queue_shift_left(current_queue)
            else:
                current_queue[0] = (current_queue[0][0]-1, current_queue[0][1])

        for i, core_queue in enumerate(queue_copy):
            pe["queue"][i] = core_queue

        self.__remove_unused_cores_cloud(pe, deleting_queues_on_pe)

    def __remove_unused_cores_cloud(self, pe, core_list):
        for i, item in enumerate(core_list):
            del list(pe["queue"])[item - i]
            del list(pe["occupiedCores"])[item - i]
            del list(pe["energyConsumption"])[item - i]

    def __task_finished(self, task_ID):
        task = self.database.get_task(task_ID)
        job_ID = task["job_id"]
        task_suc = task['successors']

        for t in task_suc:
            self.database.task_pred_dec(t)

            if self.database.get_task(t)['pred_count'] == 0:
                job = self._jobs[job_ID]
                if t in job['remainingTasks']:
                    self.preprocessor.queue.append(t)
                try:
                    self.preprocessor.wait_queue.remove(t)
                except ValueError:
                    pass

        try:
            # Convert manager lists to regular lists for modification

            self._jobs[job_ID]["finishedTasks"].append(task_ID)
            if task_ID in self._jobs[job_ID]["runningTasks"]:
                self._jobs[job_ID]["runningTasks"].remove(task_ID)

            # Convert lists back to manager lists if needed
            # self._jobs[job_ID]["finishedTasks"] = finished_tasks
            # self._jobs[job_ID]["runningTasks"] = running_tasks

        except Exception as e:
            print(f"Error processing task {task_ID} for job {job_ID}: {e}")
            raise

    def __update_occupied_cores(self, pe, pe_ID):
        for core_index, core in enumerate(list(pe["occupiedCores"])):
            if is_core_free(pe["queue"][core_index]):
                list(pe["occupiedCores"])[core_index] = 0
                # set default energy cons for idle cores
                list(pe["energyConsumption"])[core_index] = self.database.get_device(pe_ID)["powerIdle"][
                    core_index]
            else:
                list(pe["occupiedCores"])[core_index] = 1

####### UTILITY #######


####### UTILITY #######
def find_place(pe, core_i):
    if pe["type"] == "cloud":
        list(pe["queue"]).append([])
        list(pe["queue"])[-1].append((0, -1))
        list(pe["energyConsumption"]).append(0)
        list(pe["occupiedCores"]).append(1)
        return 0, len(pe["queue"]) - 1
    else:
        for i, slot in enumerate(list(pe["queue"])[core_i]):
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
