import time

from data.db import Database
from environment.agent import Agent
from environment.state import State
from environment.window_manager import WindowManager
from data.configs import environment_config

class Environment:

    def __init__(self):
        self.agent = Agent()
        self.state = State()
        self.db = Database().load()
        self.window_generator = WindowManager(self.db.get_all_jobs())
        self.cycle_wait = environment_config["environment"]["cycle"]
        self.__runner_flag = False

    def run(self):
        self.state.initialize()
        while self.__runner_flag:
            # state.get() : return states and jobs as dictionaries
            self.agent.take_action(self.state.get())
            self.state.environment_update(self.window_generator.get_window())
            time.sleep(self.cycle_wait)
