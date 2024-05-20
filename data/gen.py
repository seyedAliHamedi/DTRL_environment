import pandas as pd
import numpy as np
from data.configs import devices_config, jobs_config, environment_config
import os
import time
import ast


class Generator:
    _task_id_counter = 0
    _device_id_counter = 0
    _job_id_counter = 0
    _devices_path = os.path.join(os.path.dirname(__file__), "resources","devices.csv")
    _job_path = os.path.join(os.path.dirname(__file__),"resources", "jobs.csv")
    _tasks_path = os.path.join(os.path.dirname(__file__),"resources", "tasks.csv")

    @classmethod
    def get_devices(cls, file_path=_devices_path):
        if os.path.exists(file_path):
            devices = pd.read_csv(file_path)
            devices["voltages_frequencies"] = devices["voltages_frequencies"].apply(lambda x: ast.literal_eval(x))
            devices["capacitance"] = devices["capacitance"].apply(
                lambda x: ast.literal_eval(x)
            )
            devices["powerIdle"] = devices["powerIdle"].apply(
                lambda x: ast.literal_eval(x)
            )
            devices["acceptableTasks"] = devices["acceptableTasks"].apply(
                lambda x: ast.literal_eval(x)
            )
            return devices
        else:
            return Generator._generate_device()

    # generate random Processing element attributes based on the bounds and ranges defined in the config

    #   frequency in KHZ
    #   voltage in Volt
    #   capacitance in nano-Farad
    #   powerIdle in Watt
    #   ISL in percentage
    #   battery capacity in W*micro-second : 36000 Ws - Equivalent to 36000*10^3 W*milli-second, 10Wh or

    @classmethod
    def _generate_device(cls, config=devices_config,file_path=_devices_path):
        devices_data = []
        for type in ("iot", "mec", "cloud"):
            config = devices_config[type]

            for _ in range(config["num_devices"]):
                cpu_cores = (
                    -1
                    if config["num_cores"]
                    == -1 else int(np.random.choice(config["num_cores"]))
                )
                device_info = {
                    "id": Generator._device_id_counter,
                    "type": type,
                    "num_cores": cpu_cores,
                    "voltages_frequencies": [
                        [
                            config["voltage_frequencies"][i]
                            for i in np.random.choice(6, size=3, replace=False)
                        ]
                        for _ in range(cpu_cores)
                    ],
                    "ISL": (
                        -1
                        if config["isl"] == -1
                        else np.random.uniform(config["isl"][0], config["isl"][1])
                    ),
                    "capacitance": [
                        np.random.uniform(
                            config["capacitance"][0], config["capacitance"][1]
                        )
                        * 1e-9
                        for _ in range(cpu_cores)
                    ],
                    "powerIdle": [
                        float(np.random.choice(config["powerIdle"])) * 1e-6
                        for core in range(cpu_cores)
                    ],
                    "battery_capacity": (
                        -1
                        if config["battery_capacity"] == -1
                        else np.random.uniform(
                            config["battery_capacity"][0], config["battery_capacity"][1]
                        )
                    ),
                    "error_rate": np.random.uniform(
                        config["error_rate"][0], config["error_rate"][1]
                    ),
                    "acceptableTasks":list(np.random.choice(
                        jobs_config["task"]["task_kinds"],
                        size=np.random.randint(2, 5),
                        replace=False,
                    )),
                    "handleSafeTask": int(
                        np.random.choice(
                            [0, 1], p=[config["safe"][0], config["safe"][1]]
                        )
                    ),
                    "maxQueue": config["maxQueue"],

                }
                Generator._device_id_counter += 1
                devices_data.append(device_info)

        devices = pd.DataFrame(devices_data)
        devices['name'] = devices.apply(lambda row: row['type'] + str(row.name), axis=1)
        devices.to_csv(file_path)
        return devices

    @classmethod
    def get_jobs(cls, file_path_jobs=_job_path, file_path_tasks=_tasks_path):
        if os.path.exists(file_path_jobs):
            jobs = pd.read_csv(file_path_jobs)
            jobs["tasks_ID"] = jobs["tasks_ID"].apply(lambda x: ast.literal_eval(x))
            jobs["heads"] = jobs["heads"].apply(lambda x: ast.literal_eval(x))
            jobs["tails"] = jobs["tails"].apply(lambda x: ast.literal_eval(x))
            jobs["tree"] = jobs["tree"].apply(lambda x: ast.literal_eval(x))
            tasks = pd.read_csv(file_path_tasks)
            tasks["predecessors"] = tasks["predecessors"].apply(lambda x: ast.literal_eval(x))
            return jobs, tasks
        else:
            return Generator._generate_jobs()

    @classmethod
    def _generate_jobs(cls, config=jobs_config,file_path_jobs=_job_path, file_path_tasks=_tasks_path):

        max_deadline = config["max_deadline"]
        max_task_per_depth = config["max_task_per_depth"]
        max_depth = config["max_depth"]

        tasks_data = []
        jobs_data = []
        for i in range(config["num_jobs"]):
            # generate jobs based on job attributes
            task_list = []
            head_tasks_list = []
            tail_tasks_list = []
            head_count = np.random.randint(1, max_task_per_depth + 1)
            depth = np.random.randint(1, max_depth + 1)
            tail_count = np.random.randint(1, max_task_per_depth + 1)

            # generate tasks in three steps:
            #   1- head tasks
            #   2- middle tasks
            #   3- tail tasks
            for j in range(head_count):
                new_head_task = Generator._generate_random_task(i, True, False)
                task_list.append(new_head_task)
                head_tasks_list.append(new_head_task)

            last_depth_task_list = []
            for task in task_list:
                last_depth_task_list.append(task)

            for j in range(depth):
                current_depth_task_list = []
                current_depth_task_count = np.random.randint(1, max_task_per_depth + 1)
                for j in range(current_depth_task_count):
                    predecessors = Generator._choose_random_tasks(last_depth_task_list)
                    new_task = Generator._generate_random_task(i, False, False, predecessors)
                    task_list.append(new_task)
                    current_depth_task_list.append(new_task)
                last_depth_task_list = current_depth_task_list

            for j in range(tail_count):
                predecessors = Generator._choose_random_tasks(last_depth_task_list)
                new_task = Generator._generate_random_task(i, False, True, predecessors)
                tail_tasks_list.append(new_task)
                task_list.append(new_task)
            deadline = np.random.randint(1, max_deadline)

            # set the tasks and update the job.tasks_list

            job = {
                "id": i,
                "task_count": len(task_list),
                "tasks_ID": [task["id"] for task in task_list],
                "heads": [
                    task["id"] for task in head_tasks_list if task["is_head"]
                ],
                "tails": [
                    task["id"] for task in tail_tasks_list if task["is_tail"]
                ],
                "tree": [(task["id"], task["predecessors"]) for task in task_list],
                "deadline": deadline,
            }
            jobs_data.append(job)

            # add tasks of this job to big data of all tasks
            for task in task_list:
                tasks_data.append(task)

        jobs = pd.DataFrame(jobs_data)
        tasks = pd.DataFrame(tasks_data)
        jobs.to_csv(file_path_jobs, index=False)
        tasks.to_csv(file_path_tasks, index=False)
        return jobs, tasks

    @classmethod
    def _choose_random_tasks(cls, task_list):
        # function to select random tasks from a certain depth
        # to act as predecessors for the tasks in the next level
        copy_task_list = task_list.copy()
        max_predecessors = len(copy_task_list)
        predecessors = []
        amount = np.random.randint(1, max_predecessors + 1)
        for i in range(amount):
            chosen_task = np.random.randint(0, len(copy_task_list))
            predecessors.append(copy_task_list[chosen_task]["id"])
            copy_task_list.pop(chosen_task)
        return predecessors

    @classmethod
    def _generate_random_task(cls, job_id, is_head, is_tail, predecessors=[], config=jobs_config["task"]):
        task_id = Generator._task_id_counter
        Generator._task_id_counter += 1
        # generate tasks based on the attribute ranges and bounds defind in the config file
        input_size = np.random.randint(config["input_size"][0], config["input_size"][1])
        output_size = np.random.randint(config["output_size"][0], config["output_size"][1])
        task_kind = np.random.choice(config["task_kinds"])
        safe = int(np.random.choice([0, 1], p=[config["safe_measurement"][0],
                                               config["safe_measurement"][1]]))
        computational_load = np.random.randint(
            config["computational_load"][0],
            config["computational_load"][1],
        )
        return {
            "id": task_id,
            "job_id": job_id,
            "computational_load": computational_load,
            "input_size": input_size,
            "output_size": output_size,
            "predecessors": predecessors,
            "is_safe": safe,
            "task_kind": task_kind,
            "is_head": is_head,
            "is_tail": is_tail,
            "execution_time": -1,
            "energy_consumed": -1,
            "status": "NOT_REGISTERED",
        }
