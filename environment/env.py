import time
from data.db import Database
from environment.agent import Agent
from environment.state import State
from utilities.monitor import Monitor
from environment.window_manager import Preprocessing, WindowManager
from data.configs import environment_config


class Environment:

    def __init__(self, n_iterations):
        self.n_iterations = n_iterations
        # ! important load db first
        Database().load()
        Monitor().init(n_iterations)
        State().initialize()
        self.cycle_wait = environment_config["environment"]["cycle"]
        self.__runner_flag = True

    def run(self):
        iteration = 0
        while iteration <= self.n_iterations:
            starting_time = time.time()

            WindowManager().run()
            State().update(iteration)
            Preprocessing().run()
            Agent().run()

            # Monitor logging
            Monitor().set_env_log(State().get(), WindowManager().get_log(), Preprocessing().get_log(), iteration)

            # Calculating time passed in iteration and saving log
            time_len = time.time() - starting_time

            # Calculate sleeping time
            sleeping_time = self.cycle_wait - time_len
            if sleeping_time < 0:
                sleeping_time = 0
                Monitor().add_time(time_len, iteration)
            else:
                Monitor().add_time(self.cycle_wait, iteration)
            time.sleep(sleeping_time)

            iteration += 1

        Monitor().save_logs()
