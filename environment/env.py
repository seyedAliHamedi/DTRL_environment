import time

from data.db import Database
from environment.agent import Agent
from environment.state import State

class Environment:

    def __init__(self):
        self.agent = Agent()
        self.state = State()
        self.window_generator = ()
        self.db = Database().load()
        self.cycle_wait = 0.001
        self.__runner_flag = False

    def run(self):
        self.state.initialize()
        while self.__runner_flag:
            # state.get() : return states and jobs as dictionaries
            self.agent.take_action(self.state.get())
            self.state.environment_update()
            time.sleep(self.cycle_wait)

    