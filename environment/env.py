import multiprocessing
import sys
import threading
import time
import traceback
from data.db import Database
from environment.a3c.actor_critic import ActorCritic
from environment.a3c.agent import Agent
from environment.a3c.shared_adam import SharedAdam
from environment.state import State
from utilities.monitor import Monitor
from utilities.memory_monitor import MemoryMonitor
from environment.window_manager import WindowManager
from environment.pre_processing import Preprocessing
from data.configs import environment_config, monitor_config, agent_config
import torch.multiprocessing as mp


class Environment:

    def __init__(self, n_iterations, display):
        self.n_iterations = n_iterations
        self.display = display
        self.memory_monitor = MemoryMonitor()
        self.mem_monitor_thread = threading.Thread(
            target=self.memory_monitor.run)
        # ! important load db first
        Database().load()
        Monitor().init(n_iterations)

        self.state = State(self, display)
        self.window_manager = WindowManager(self)
        self.pre_processing = Preprocessing(self)
        self.agent = Agent()
        self.agent.init(self, display)
        self.cycle_wait = environment_config["environment"]["cycle"]

    def run(self):
        # TODO : decide worker.run / multithread/ !!!!! multiprocess pytorch --> .start() & .join()
        global_actor_critic = ActorCritic(
            input_dims=5, n_actions=len(Database.get_all_devices()))
        global_actor_critic.share_memory()
        optim = SharedAdam(global_actor_critic.parameters())

        iteration = 0
        try:
            self.mem_monitor_thread.start()
            while iteration <= self.n_iterations:
                if iteration % 100 == 0:
                    print(f"iteration : {iteration}")

                starting_time = time.time()
                self.window_manager.run()
                self.state.update()
                self.pre_processing.run()
                self.agent.run()
                time_len = time.time() - starting_time

                # Calculate sleeping time

                self.sleep(time_len, iteration)

                self.monitor_log(iteration)
                iteration += 1

        except KeyboardInterrupt as e:
            print("Interrupted")
        except Exception as e:
            print("Caught an unexpected exception:")
            traceback.print_exc()
        finally:
            Monitor().save_logs()
            print("Jobs Done:", self.state.jobs_done)
            print("WaitQueue Length", len(self.pre_processing.wait_queue))
            self.memory_monitor.stop()

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
                                  self.pre_processing.get_log(), iteration)
