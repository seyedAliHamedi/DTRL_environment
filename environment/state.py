from data.gen import Generator

class State:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._PEs = []
            cls._instance._jobs = []
        return cls._instance

    def initialize(self):
        self.init_PEs(Generator.get_devices())
        self.init_jobs(Generator.get_jobs()[0])

    def init_PEs(self, PEs):
        dict_PEs = PEs.to_dict(orient='records')
        for pe in dict_PEs:
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
        dict_jobs = jobs.to_dict(orient='records')
        for job in dict_jobs:
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

    def environment_update(self, instructions):

        new_jobs = instructions["jobs"]
        self.init_new_jobs(new_jobs)

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
