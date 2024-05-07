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
                    "queue": [
                        [(0, -1) for core in range(pe["num_cores"])]
                        for _ in range(pe["maxQueue"])
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
                    "assignedTasks": [],
                    "remainingTasks": job["tasks_ID"],
                    "remainingDeadline": job["deadline"],
                }
            )

    def get(self):
        return self._jobs, self._PEs

    ####### ENVIRONMENT #######

    def environment_update(self, job_window):

        job_window
        self.__update_PEs_queue()
        self.__update_energy_consumption()
        self.__update_batteries_capp()

    def __update_batteries_capp(self):
        for pe in self._PEs:
            pe["batteryLevel"] -= sum(pe["energyConsumption"])
        return

    def __update_energy_consumption(self):
        for pe in self._PEs:
            pass
        return

    def __update_PEs_queue(self):
        return

    def __update_active_jobs(self, jobs):
        pass
