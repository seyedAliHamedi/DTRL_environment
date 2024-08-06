import matplotlib.pyplot as plt
import os
import json
import threading
import time
import traceback

from environment.a3c.actor_critic import ActorCritic
from environment.a3c.agent import Agent
from environment.a3c.shared_adam import SharedAdam
from environment.state import State
from utilities.mini_monitor import MiniMonitor
from utilities.memory_monitor import MemoryMonitor
from data.configs import environment_config, monitor_config, agent_config

import torch.multiprocessing as mp


class Environment:

    def __init__(self, n_iterations, display):
        self.n_iterations = n_iterations
        self.memory_monitor = MemoryMonitor()
        self.mem_monitor_thread = threading.Thread(
            target=self.memory_monitor.run)
        self.cycle_wait = environment_config["environment"]["cycle"]
        self.__runner_flag = True
        self.__worker_flags = []

        manager = mp.Manager()
        self.monitor = MiniMonitor(n_iterations, manager)
        lock = mp.Lock()
        self.manager = manager
        self.state = State(display, manager=manager, lock=lock)
        self.db = self.state.database
        self.preprocessor = self.state.preprocessor
        self.window_manager = self.state.window_manager
        self.time_log = []

    def run(self):
        global_actor_critic = ActorCritic(
            input_dims=5, n_actions=len(self.db.get_all_devices()))
        global_actor_critic.share_memory()
        optim = SharedAdam(global_actor_critic.parameters())
        workers = []
        barrier = mp.Barrier(agent_config['multi_agent'] + 1)

        self.state.update(self.manager)
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
                if iteration % 500 == 0:
                    self.make_agents_plots()

                starting_time = time.time()

                self.state.update(self.manager)
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
            self.monitor.save_logs()
            self.save_time_log()
            self.make_agents_plots()
            print(self.state.jobs_done)
            print(len(self.preprocessor.wait_queue))

            for worker in workers:
                worker.stop()

            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
                    worker.join()

            self.memory_monitor.stop()

    def sleep(self, time_len, iteration):
        sleeping_time = self.cycle_wait - time_len
        if sleeping_time < 0:
            sleeping_time = 0
            if monitor_config['settings']['time']:
                self.monitor.add_time(time_len, iteration)
                self.time_log.append(time_len)
        else:
            if monitor_config['settings']['time']:
                self.monitor.add_time(self.cycle_wait, iteration)
                self.time_log.append(self.cycle_wait)
        time.sleep(sleeping_time)

    def monitor_log(self, iteration):
        if monitor_config['settings']['main']:
            self.monitor.set_env_log(self.state.get(), self.window_manager.get_log(),
                                     self.preprocessor.get_log(), iteration)

    def save_time_log(self):
        y_values = self.time_log
        plt.figure(figsize=(10, 5))
        plt.plot(y_values, marker='o', linestyle='-')
        plt.axhline(y=environment_config['environment']['cycle'],
                    color='red', linestyle='--', label='-set-cycle-time')
        plt.title("Sleeping time on each iteration")
        plt.xlabel("iteration")
        plt.ylabel("sleeping time")
        plt.grid(True)
        plt.legend()
        path = monitor_config['paths']['time']['plot']
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        with open(monitor_config['paths']['time']['summery'], 'w') as f:
            json.dump(self.time_log, f, indent=4)

    def make_agents_plots(self, path=monitor_config['paths']['agent']['plots']):
        print("~~~~~~ MAKING AGENT PLOTS ~~~~~~~")
        filtered_data = {k: v for k, v in self.state.agent_log.items() if v}
        time_list = [v["time"] for v in filtered_data.values()]
        energy_list = [v["energy"] for v in filtered_data.values()]
        fails_list = [v["fails"] for v in filtered_data.values()]
        reward_list = [v["reward"] for v in filtered_data.values()]
        loss_list = [v["loss"] for v in filtered_data.values()]

        fig, axs = plt.subplots(3, 2, figsize=(20, 15))
        axs[0, 0].plot(loss_list,
                       label='Loss', color="blue", marker='o')
        axs[0, 0].set_title('Loss')
        axs[0, 0].legend()

        # Plot for reward
        axs[0, 1].plot(reward_list,
                       label='Reward', color="cyan", marker='o')
        axs[0, 1].set_title('Reward')
        axs[0, 1].legend()

        # Plot for time
        axs[1, 0].plot(time_list,
                       label='Time', color="red", marker='o')
        axs[1, 0].set_title('Time')
        axs[1, 0].legend()

        # Plot for energy
        axs[1, 1].plot(energy_list,
                       label='Energy', color="green", marker='o')
        axs[1, 1].set_title('Energy')
        axs[1, 1].legend()

        # Plot for fail
        axs[2, 0].plot(fails_list,
                       label='Fail', color="purple", marker='o')
        axs[2, 0].set_title('Fail')
        axs[2, 0].legend()

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Save the plots to an image file
        plt.savefig(path)
