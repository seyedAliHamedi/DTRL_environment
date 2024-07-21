devices_config = {
    "iot": {
        "num_devices": 10,
        "num_cores": [4, 8, 16],
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
        "safe": (0.1, 0.9),
        "maxQueue": 5
    },
    "mec": {
        "num_devices": 0,
        "num_cores": [16, 32, 64],
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
        "safe": (0.25, 0.75),
        "maxQueue": 5

    },
    "cloud": {
        "num_devices": 0,
        "num_cores": -1,
        "voltage_frequencies": ((2.8e9, 13.85), (3.9e9, 24.28)),
        "isl": -1,
        "capacitance": -1,
        "powerIdle": 0,
        "battery_capacity": -1,
        "error_rate": (0.10, 0.15),
        "safe": (1, 0),
        "maxQueue": 1
    },
}

jobs_config = {
    "num_jobs": 10000,
    "max_deadline": 2000,
    "max_task_per_depth": 5,
    "max_depth": 3,
    "task": {
        "input_size": [1000, 1000000],
        "output_size": [1000, 1000000],
        "computational_load": [1000, 1000000],
        "safe_measurement": [0.95, 0.05],
        "task_kinds": [1, 2, 3, 4],
    },
}
environment_config = {
    "generator": {"iot": 1, "mec": 0, "cloud": 0, "jobs_count": 100},
    "window": {"size": 10, "max_jobs": 5, "clock": 3},
    "environment": {"cycle": 0.01}

}
monitor_config = {
    'paths': {
        'time': {
            'plot': './logs/simulation/time_plot.png',
            'summery': './logs/simulation/summery.csv',
        },
        'main': {
            'pes': './logs/main/pe.csv',
            'jobs': './logs/main/job.csv',
            'window': './logs/main/window.csv',
            'preprocessing': './logs/main/preprocessing.csv',
        },
        'summary': './logs/summary.txt',
        'agent': {
            'summary': './logs/agent/summary.txt',
        }
    }
}

agent_config = {
    'learning_mod': 1,
    'alpha': 1,
    'beta': 1,
    'learning_rate': 0.1,
    'gamma': 0.99
}

summary_log_string = '|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||'
