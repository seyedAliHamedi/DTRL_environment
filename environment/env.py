import time

from data.db import Database
from environment.agent import Agent
from environment.state import State
from environment.window_manager import WindowManager
from data.configs import environment_config

class Environment:

    def __init__(self):
        self.agent = Agent()
        # ! important load db first
        Database().load()
        self.state = State().initialize()
        self.window_generator = WindowManager()
        self.cycle_wait = environment_config["environment"]["cycle"]
        self.__runner_flag = True

    def run(self):
        count = 0
        while self.__runner_flag:
            # state.get() : return states and jobs as dictionaries
            self.agent.take_action(self.state.get())
            self.state.environment_update(self.window_generator.get_window())
            time.sleep(self.cycle_wait)
Environment().run()