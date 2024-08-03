import multiprocessing
import threading
import time
import traceback

from environment.a3c.actor_critic import ActorCritic
from environment.a3c.agent import Agent
from environment.a3c.shared_adam import SharedAdam
from environment.state import State
from utilities.monitor import Monitor
from utilities.memory_monitor import MemoryMonitor
from data.configs import environment_config, monitor_config, agent_config

import torch.multiprocessing as mp


class Environment:

    def __init__(self, n_iterations, display):
        self.n_iterations = n_iterations
        self.memory_monitor = MemoryMonitor()
        self.mem_monitor_thread = threading.Thread(
            target=self.memory_monitor.run)
        Monitor().init(n_iterations)
        self.cycle_wait = environment_config["environment"]["cycle"]
        self.__runner_flag = True
        self.__worker_flags = []

        manager = mp.Manager()
        lock = mp.Lock()
        self.state = State(manager=manager, lock=lock)
        self.state.initialize(display=display)
        self.db = self.state.database
        self.preprocessor = self.state.preprocessor
        self.window_manager = self.state.window_manager

    def run(self):
        global_actor_critic = ActorCritic(
            input_dims=5, n_actions=len(self.db.get_all_devices()))
        global_actor_critic.share_memory()
        optim = SharedAdam(global_actor_critic.parameters())
        workers = []
        barrier = mp.Barrier(agent_config['multi_agent'] + 1)

        for i in range(agent_config['multi_agent']):
            worker = Agent(
                name=f'worker_{i}', global_actor_critic=global_actor_critic, optimizer=optim, barrier=barrier, shared_state=self.state)
            workers.append(worker)
            worker.start()

        iteration = 0
        try:
            self.mem_monitor_thread.start()
            while iteration <= self.n_iterations:
                if iteration % 10 == 0:
                    print(f"iteration : {iteration}")

                starting_time = time.time()
                self.window_manager.run()
                self.state.update()
                self.preprocessor.run()
                print("-------- ", self.state.get_agent_queue2())
                time.sleep(1)
                # for worker in workers:
                #     worker.run()
                # [w.join() for w in workers]

                # Monitor logging

                # Calculate sleeping time
                barrier.wait()
                time_len = time.time() - starting_time
                self.sleep(time_len, iteration)

                self.monitor_log(iteration)
                iteration += 1

        except KeyboardInterrupt:
            print("Interrupted")
        except Exception as e:
            print("Caught an unexpected exception:")
            traceback.print_exc()
        finally:
            Monitor().save_logs()
            print(self.state.jobs_done)
            print(len(self.preprocessor.wait_queue))

            for worker in workers:
                worker.stop()
            for worker in workers:
                worker.join()
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
            Monitor().set_env_log(self.state.get(), self.window_manager.get_log(),
                                  self.preprocessor.get_log(), iteration)
