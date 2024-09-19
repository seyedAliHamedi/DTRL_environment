import time
import traceback

from data.db import Database
from environment.AgentWrapper import AgentWrapper
from environment.state import State
from utilities.monitor import Monitor
from environment.window_manager import Preprocessing, WindowManager
from data.configs import environment_config, monitor_config


class Environment:

    def __init__(self, n_iterations, display):
        self.n_iterations = n_iterations
        self.display = display
        # ! important load db first
        self.db = Database()
        Monitor().init(n_iterations)
        State(db=self.db).initialize(display)
        Preprocessing(db=self.db)
        WindowManager(db=self.db)
        self.cycle_wait = environment_config["environment"]["cycle"]
        self.__runner_flag = True

    def run(self):
        agent = AgentWrapper(len(self.db.get_all_devices()), 1, 8, self.db)
        iteration = 0
        try:
            while iteration <= self.n_iterations:
                if iteration % 100 == 0:
                    print(f"iteration : {iteration}, active_jobs:{State().get_jobs_len()}, done_jobs:{State().done_jobs}")
                    if iteration != 0:
                        # Monitor().save_logs()
                        agent.plot_logs(monitor_config['paths']['agent']['plots'])

                starting_time = time.time()

                WindowManager().run()
                State().update()
                Preprocessing().run()
                agent.run(self.display)

                time_len = time.time() - starting_time

                # Calculate sleeping time
                self.sleep(time_len, iteration)

                iteration += 1
        except KeyboardInterrupt:
            print("Interrupted")
        finally:
            Monitor().save_logs()

    def sleep(self, time_len, iteration):
        sleeping_time = self.cycle_wait - time_len
        if sleeping_time < 0:
            sleeping_time = 0
            if monitor_config['settings']['time']:
                Monitor().add_time(time_len, iteration)
        else:
            if monitor_config['settings']['time']:
                Monitor().add_time(self.cycle_wait, iteration)
        time.sleep(sleeping_time)

    def monitor_log(self, iteration):
        if monitor_config['settings']['main']:
            Monitor().set_env_log(State().get(), WindowManager().get_log(),
                                  Preprocessing().get_log(), iteration)
