import pandas as pd

from data.gen import Generator


class State:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._PEs = []
            cls._instance._jobs = []
        return cls._instance

    def init_PEs(self, PEs):
        dict_PEs = PEs.to_dict(orient='records')
        for pe in dict_PEs:
            self._PEs.append(
                {
                    "type": pe["type"],
                    "id": pe["id"],
                    "batteryLevel": pe["batteryLevel"],
                    "occupiedCores": [0 for core in range(pe["number_of_cpu_cores"])],
                    "energyConsumption": pe["powerIdle"],
                    "queue": [
                        [
                            (0, -1) for core in range(pe["number_of_cpu_cores"])
                        ]
                        for _ in range(pe["maxQueue"])
                    ]
                }
            )

    def init_new_jobs(self, jobs):
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

    
    def get_state(self):
        return pd.DataFrame(self._PEs), pd.DataFrame(self._jobs)

    ####### AGENT #######

    def agent_update(self):
        return

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


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)



State().init_new_jobs( Generator.get_jobs()[0])
State().init_PEs(Generator.get_devices())
pes_pandas, jobs_pandas = State().get_state()
print(pes_pandas)
print("///////////////")
print(jobs_pandas)
