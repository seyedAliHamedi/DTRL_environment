from data.db import Database
from data.gen import Generator


class State:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._PEs = []
            cls._instance._jobs = []
            cls._active_jobs_count = 5
        return cls._instance

    def initialize(self):
        self.init_PEs(Database.get_all_devices())
        self.init_jobs(Database.get_jobs(self._active_jobs_count))

    def init_PEs(self, PEs):
        for pe in PEs:
            self._PEs.append(
                {
                    "type": pe["type"],
                    "id": pe["id"],
                    "batteryLevel": pe["batteryLevel"],
                    "occupiedCores": [0 for core in range(pe["num_cores"])],
                    "energyConsumption": pe["powerIdle"],
                    # time,task_id
                    "queue": [
                        [(0, -1) for _ in range(pe["maxQueue"])]
                        for core in range(pe["num_cores"])
                    ],
                }
            )

    def init_jobs(self, jobs):
        for job in jobs:
            self._jobs.append(
                {
                    "id": job["id"],
                    "task_count": job["task_count"],
                    "finishedTasks": [],
                    "assignedTask": {},
                    "runningTasks": [],
                    "unScheduled": job["tasks_ID"],
                    "remainingDeadline": job["deadline"],
                }
            )

    def get(self):
        return self._jobs, self._PEs

    ####### ENVIRONMENT #######

    def environment_update(self, job_window):

        print(f"new_job_window: {job_window}")

        # process 1
        self.__update_jobs()
        # process 2
        self.__update_PEs()
        # TODO pop assigned task

    ### PROCESS 1 ###

    def __update_jobs(self):
        # update remaining_task
        # update running_tasks
        # update finished_tasks
        # update deadline
        pass

    def __update_remaining_tasks(self):
        pass

    def __update_running_tasks(self):
        pass

    def __update_finished_tasks(self):
        pass

    def __update_deadlines(self):
        pass

    ### PROCESS 2 ###

    def __update_PEs(self):
        self.__update_PEs_queue()
        self.__update_occupied_cores()
        self.__update_batteries_capp()
        self.__update_energy_consumption()

    def __update_batteries_capp(self):
        # time
        for pe in self._PEs:
            pe["batteryLevel"] -= sum(pe["energyConsumption"])

    def __update_energy_consumption(self, pe_index, core_index, dvfs):
        # agent
        pe = Database().get_device(self._PEs[pe_index]["id"])
        freq, vol = pe["voltages_frequencies"][core_index][dvfs]
        capacitence = pe["capacitance"][core_index]
        self._PEs[pe_index][core_index] = capacitence*(vol*vol)*freq

        # time
        for pe_dict in self._PEs:
            for index in range(pe_dict["energyConsumption"]):
                if pe_dict["occupiedCores"][index] == 0:
                    self._PEs[pe_index][core_index] = pe["powerIdle"][core_index]

    def __update_PEs_queue(self, task_id , pe_index, core_index, dvfs):
        # agent
        current_quque = self._PEs[pe_index]["queue"][core_index]
        pe = Database().get_device(self._PEs[pe_index]["id"])
        task = Database().get_task(task_id)
        for elem in current_quque :
            if elem != (0,-1):
                continue
            freq, vol = pe["voltages_frequencies"][core_index][dvfs]
            elem = (task["computational_load"] / freq, task_id)
            break

        # time
        element = current_quque[0]
        if element[0]> 1:
            element[0] -=1
        else:
            element.pop(0)
            element.append((0,-1))

    def __update_occupied_cores(self):
        # both
        for pe_index in range(len(self._PEs)):
            occupiedCores = self._PEs[pe_index]["occupiedCores"]
            for core_index in range(len(occupiedCores)):
                queue = self._PEs[pe_index]["queue"][core_index]
                if queue[0]==(0,-1):
                    occupiedCores[core_index] = 0
                else:
                    occupiedCores[core_index] = 1
