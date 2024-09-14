devices_config = {
    "iot": {
        "num_devices": 7,
        "num_cores": [2, 4, 8],
        "voltage_frequencies": [
            (10e6, 1.8),
            (20e6, 2.3),
            (40e6, 2.7),
            (80e6, 4.0),
            (160e6, 5.0),
        ],
        "isl": (0.1, 0.2),
        # capacitance in nano-Farad --> * 1e-9
        "capacitance": (0.2, 0.3),
        # powerIdle in micro-Watt --> * 1e-6
        "powerIdle": [800, 900, 1000],
        # battery_capacity in Watt-second
        "battery_capacity": (36, 41),
        "error_rate": (0.01, 0.06),
        "safe": (0.25, 0.75),
        "maxQueue": 5
    },
    "mec": {
        "num_devices": 0,
        "num_cores": [8, 16, 32],
        "voltage_frequencies": [
            (600 * 1e6, 0.8),
            (750 * 1e6, 0.825),
            (1000 * 1e6, 1.0),
            (1500 * 1e6, 1.2),
        ],
        "isl": -1,
        # capacitance in nano-Farad --> * 1e-9
        "capacitance": (1.5, 2),
        # powerIdle in micro-Watt --> * 1e-6
        "powerIdle": [550000, 650000, 750000],
        "battery_capacity": -1,
        "error_rate": (0.5, 0.11),
        "safe": (0.5, 0.5),
        "maxQueue": 1

    },
    "cloud": {
        "num_devices": 1,
        "num_cores": 128,
        # TODO cloud , please correct these numbers
        "voltage_frequencies": ((2.8e9, 13.85e-6), (3.9e9, 24.28e-6), (5e9, 36e-6)),
        "isl": -1,
        "capacitance": (3, 5),
        "powerIdle": [0],
        "battery_capacity": -1,
        "error_rate": (0.10, 0.15),
        "safe": (1,0),
        "maxQueue": 1
    },
}

jobs_config = {
    "num_jobs": 10000,
    "max_deadline": 2000,
    "max_task_per_depth": 2,
    "max_depth": 2,
    "task": {
        # input_size ,output_size ,computational_load in MB --> * 10^6
        "input_size": [1, 1001],
        "output_size": [1, 1001],
        "computational_load": [1, 1001],
        "safe_measurement": [0.8, 0.2],
        "task_kinds": [1, 2, 3, 4],
    },
}

environment_config = {
    "multi_agent": 15,
    "time_out_counter": 150,
    "window": {"size": 100, "max_jobs": 20, "clock": 10},
    "environment": {"cycle": 0.01, "anomaly_th": 5},
}

monitor_config = {
    'paths': {
        'time': {
            'plot': './logs/simulation/time_plot.png',
            'summery': './logs/simulation/summery.csv',
        },
        'summary': './logs/summary.txt',
        'agent': {
            'summary': './logs/agent/summary.txt',
            'plots': './logs/agent/plots.png'
        },
        'memory': './logs/simulation/memory.png',
    }
}
