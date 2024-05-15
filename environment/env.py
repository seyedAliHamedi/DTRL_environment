import time

from data.db import Database
from environment.agent import Agent
from environment.state import State
from environment.utilities.window_manager import PreProccesing, WindowManager
from data.configs import environment_config


class Environment:

    def __init__(self):
        #! important load db first
        Database().load()
        State().initialize()
        self.cycle_wait = environment_config["environment"]["cycle"]
        self.__runner_flag = True

    def run(self):
        pre_processing = PreProccesing()
        while self.__runner_flag:
            WindowManager().run()
            PreProccesing().run()
            Agent().run()
            State().update()
            time.sleep(self.cycle_wait)
