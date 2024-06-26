devices_config = {
    "iot": {
        "num_devices": 3,
        "num_cores": [4, 2],
        "voltage_frequencies": [
            (1e6, 1.8),
            (2e6, 2.3),
            (4e6, 2.7),
            (8e6, 4.0),
            (16e6, 5.0),
            (32e6, 6.5),
        ],
        "isl": (0.1, 0.2),
        # capacitance in nano-Farad --> * 1e-9
        "capacitance": (2, 3),
        # powerIdle in micro-Watt --> * 1e-6
        "powerIdle": [800, 900, 1000],
        # battery_capacity in Watt-second
        "battery_capacity": (36000, 40000),
        "error_rate": (0.005, 0.01),
        "safe": (0.1, 0.9),
        "maxQueue": 5
    },
    "mec": {
        "num_devices": 0,
        "num_cores": [16, 32, 64],
        "voltage_frequencies": [
            (6 * 1e8, 0.8),
            (7.5 * 1e8, 0.825),
            (10 * 1e8, 1.0),
            (15 * 1e8, 1.2),
            (30 * 1e8, 2),
            (40 * 1e8, 3.1),
        ],
        "isl": -1,
        # capacitance in nano-Farad --> * 1e-9
        "capacitance": (1.5, 2),
        # powerIdle in micro-Watt --> * 1e-6
        "powerIdle": [600, 700, 900],
        "battery_capacity": -1,
        "error_rate": (0.01, 0.015),
        "safe": (0.25, 0.75),
        "maxQueue": 50
    },
    "cloud": {
        "num_devices": 0,
        "num_cores": -1,
        "voltage_frequencies": ((50000, 13.85), (80000, 24.28)),
        "isl": -1,
        "capacitance": -1,
        "powerIdle": 0,
        "battery_capacity": -1,
        "error_rate": (0.01, 0.015),
        "safe": (0.4, 0.6),
        "maxQueue": 100
    },
}

jobs_config = {
    "num_jobs": 100,
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
