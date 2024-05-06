devices = {
    "iot": {
        "num_devices": 10,
        "num_cores": [1, 2, 4],
        "voltage_frequencies": [
            [1000, 1.8],
            [2000, 2.3],
            [4000, 2.7],
            [8000, 4.0],
            [16000, 5.0],
            [32000, 6.5]
        ],
        "isl": [0.1, 0.2],
        "capacitance": [2, 3],
        "powerIdle": [0.7, 0.8, 0.9],
        "battery_capacity": [36, 40],
        "error_rate": [0.05, 0.1]
    },
    "mec": {
        "num_devices": 10,
        "num_cores": [16, 32, 64],
        "voltage_frequencies": [
            [6000, 0.8],
            [7500, 0.825],
            [10000, 1.0],
            [15000, 1.2],
            [30000, 2],
            [40000, 3.1]
        ],
        "isl": -1,
        "capacitance": [0.0000000015, 0.000000002],
        "powerIdle": [0.00008, 0.00009, 0.0001],
        "battery_capacity": -1,
        "error_rate": [0.1, 0.15]
    },
    "cloud": {
        "num_devices": 1,
        "num_cores": -1,
        "voltage_frequencies": [
            [50000, 13.85],
            [80000, 24.28]
        ],
        "isl": -1,
        "capacitance": [0.000000002, 0.000000001],
        "powerIdle": 0,
        "battery_capacity": -1,
        "error_rate": [0.1, 0.15]
    }
},

job = {
    "max_deadline": 2000,
    "max_task_per_depth": 5,
    "max_depth": 3,

    "task": {
        "input_size": [1000, 1000000],
        "output_size": [1000, 1000000],
        "task_kinds": 4,
        "computational_load": [1000, 1000000],
        "safe_measurement": [0.95, 0.05]
    }
}

environment = {
    "generator": {
        "iot": 1,
        "mec": 0,
        "cloud": 0,
        "jobs_count": 100
    },
    "window": {
        "size": 10,
        "max_jobs": 5
    }
}
