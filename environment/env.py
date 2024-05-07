import time

from data.db import Database
from environment.state import State
from environment.utilities.window_manager import WindowManager
from data.configs import environment_config


class Environment:

    def __init__(self):
        # self.agent = Agent()
        # ! important load db first
        Database().load()
        State().initialize()
        self.window_generator = WindowManager()
        self.cycle_wait = environment_config["environment"]["cycle"]
        self.__runner_flag = True

    def run(self):
        while self.__runner_flag:
            # state.get() : return states and jobs as dictionaries
            # self.agent.take_action()
            State().environment_update(self.window_generator.get_window())
            time.sleep(self.cycle_wait)


Environment().run()
