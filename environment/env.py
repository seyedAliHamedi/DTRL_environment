import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json
import time
import traceback
import data.configs
from environment.model.actor_critic import ActorCritic
from environment.model.agent import Agent
from environment.model.utilities import SharedAdam
from environment.monitor import Monitor
from environment.state import State
from data.configs import environment_config, monitor_config

import torch.multiprocessing as mp


class Environment:

    def __init__(self, n_iterations, display):
        # numver of iterations for the environment/simulation
        self.n_iterations = n_iterations
        # the minimumm wait between iterations
        self.cycle_wait = environment_config["environment"]["cycle"]
        # the shared state and the manager for shared memory variables/dictionaries/lists
        manager = mp.Manager()
        self.state = State(display=display, manager=manager)
        self.manager = manager
        # the 3 global enteties of the shared state (preProcessor,windowManager,database)
        self.db = self.state.database
        self.monitor = Monitor()
        self.preprocessor = self.state.preprocessor
        self.window_manager = self.state.window_manager

        self.time_log = []
        self.display = display

    def run(self):
        print("Loading devices...")
        self.devices = self.db.get_all_devices()
        # define the global Actor-Critic and the shared optimizer (A3C)
        global_actor_critic = ActorCritic(input_dim=8 + len(self.devices), output_dim=len(self.devices),
                                          tree_max_depth=4, critic_input_dim=8 + len(self.devices),
                                          critic_hidden_layer_dim=128, discount_factor=0.99)
        global_actor_critic.share_memory()
        global_optimizer = SharedAdam(global_actor_critic.parameters())
        # setting up workers and their barriers
        self.workers = []
        barrier = mp.Barrier(environment_config['multi_agent'] + 1)
        # kick off the state
        print("Simulation starting...")
        self.state.update(self.manager)
        for i in range(environment_config['multi_agent']):
            # organize and start the agents
            worker = Agent(name=f'worker_{i}', global_actor_critic=global_actor_critic,
                           global_optimizer=global_optimizer, barrier=barrier, shared_state=self.state,
                           time_out_counter=data.configs.environment_config["time_out_counter"])
            self.workers.append(worker)
            worker.start()

        iteration = 0
        try:
            while iteration <= self.n_iterations:
                if iteration % 10 == 0 and iteration != 0:
                    print(f"iteration : {iteration}", len(self.state.get_jobs()))
                    if iteration % 100 == 0:
                        self.save_time_log(monitor_config['paths']['time']['plot'])
                        self.make_agents_plots()
                        self.monitor.plot()

                starting_time = time.time()
                self.state.update(self.manager)
                self.monitor.log_PEs(self.state.get_PEs())
                barrier.wait()

                time_len = time.time() - starting_time
                self.sleep(time_len)
                iteration += 1

        except Exception as e:
            print("Caught an unexpected exception:", e)
            traceback.print_exc()
        finally:
            print("Simulation Finished")
            print("Saving Logs......")
            self.save_time_log(monitor_config['paths']['time']['plot'])
            self.make_agents_plots()

            # stopping and terminating the workers
            for worker in self.workers:
                worker.stop()
            for worker in self.workers:
                if worker.is_alive():
                    worker.terminate()
                    worker.join()

            self.memory_monitor.stop()

    def sleep(self, time_len):
        # sleep for the minimum time or the time that the actual simulation took
        sleeping_time = self.cycle_wait - time_len
        if sleeping_time < 0:
            sleeping_time = 0
            self.time_log.append(time_len)
        else:
            self.time_log.append(self.cycle_wait)
        time.sleep(sleeping_time)

    def save_time_log(self, path):
        if len(self.time_log) == 0:
            return
        # saving time log gathered in the simulation
        y_values = self.time_log
        time_summary_log = {
            'total_time': sum(y_values),
            'average_time': sum(y_values) / len(y_values),
            'max_time': max(y_values),
            'epochs': len(y_values),
        }
        plt.figure(figsize=(10, 5))
        plt.plot(y_values, marker='o', linestyle='-')
        plt.axhline(y=environment_config['environment']['cycle'],
                    color='red', linestyle='--', label='-set-cycle-time')
        plt.title("Sleeping time on each iteration")
        plt.xlabel("iteration")
        plt.ylabel("sleeping time")
        plt.grid(True)
        plt.legend()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        with open(monitor_config['paths']['time']['summery'], 'w') as f:
            json.dump(time_summary_log, f, indent=2)

    def make_agents_plots(self, plot_path=monitor_config['paths']['agent']['plots']):
        # saving the agent logs gatherd from the state
        filtered_data = {k: v for k, v in self.state.agent_log.items() if v}
        time_list = [v["time"] for v in filtered_data.values()]
        energy_list = [v["energy"] for v in filtered_data.values()]
        reward_list = [v["reward"] for v in filtered_data.values()]
        loss_list = [v["loss"] for v in filtered_data.values()]

        fails_list = [v["fails"] for v in filtered_data.values()]
        safe_fails_list = [v["safe_fails"] for v in filtered_data.values()]
        kind_fails_list = [v["kind_fails"] for v in filtered_data.values()]
        queue_fails_list = [v["queue_fails"] for v in filtered_data.values()]
        battery_fails_list = [v["battery_fails"] for v in filtered_data.values()]
        iot_usage = [v["iot_usuage"] for v in filtered_data.values()]
        mec_usuage = [v["mec_usuage"] for v in filtered_data.values()]
        cc_usuage = [v["cc_usuage"] for v in filtered_data.values()]

        path_history = self.state.paths

        fig, axs = plt.subplots(5, 2, figsize=(15, 30))
        axs[0, 0].plot(loss_list,
                       label='Loss', color="blue", marker='o')
        axs[0, 0].set_title('Loss')
        axs[0, 0].legend()

        # Plot for time
        axs[0, 1].plot(time_list,
                       label='Time', color="red", marker='o')
        axs[0, 1].set_title('Time')
        axs[0, 1].legend()

        # Plot for energy
        axs[1, 0].plot(energy_list,
                       label='Energy', color="green", marker='o')
        axs[1, 0].set_title('Energy')
        axs[1, 0].legend()

        # Plot for all fail
        axs[1, 1].plot(fails_list,
                       label='ALL Fails', color="purple", marker='o')
        axs[1, 1].set_title('Fail')
        axs[1, 1].legend()

        axs[2, 0].plot(safe_fails_list,
                       label='Safe task Fail', color="purple", marker='o')
        axs[2, 0].set_title('Fail')
        axs[2, 0].legend()

        # Plot for kind fail
        axs[2, 1].plot(kind_fails_list,
                       label='Task kind Fail', color="purple", marker='o')
        axs[2, 1].set_title('Fail')
        axs[2, 1].legend()

        # Plot for queue fail
        axs[3, 0].plot(queue_fails_list,
                       label='Queue full Fail', color="purple", marker='o')
        axs[3, 0].set_title('Fail')
        axs[3, 0].legend()

        # Plot for battery fail
        axs[3, 1].plot(battery_fails_list,
                       label='Battery punish', color="purple", marker='o')
        axs[3, 1].set_title('Fail')
        axs[3, 1].legend()

        axs[4, 0].plot(iot_usage, label='IoT Usage', color='blue', marker='o')
        axs[4, 0].plot(mec_usuage, label='MEC Usage', color='orange', marker='x')
        axs[4, 0].plot(cc_usuage, label='Cloud Usage', color='green', marker='s')
        axs[4, 0].set_title('Devices Usage History')
        axs[4, 0].set_xlabel('Epochs')
        axs[4, 0].set_ylabel('Usage')
        axs[4, 0].legend()
        axs[4, 0].grid(True)

        # Heatmap for path history
        # print(path_history)
        if path_history and len(path_history) > 0:

            output_classes = make_paths(len(path_history[0][0]))
            path_counts = np.zeros((len(path_history), len(output_classes)))

            for epoch in range(len(path_history)):
                epoch_paths = path_history[epoch]

                for path in epoch_paths:
                    path_index = output_classes.index(path)
                    path_counts[epoch, path_index] += 1

            sns.heatmap(path_counts, cmap="YlGnBu",
                        xticklabels=output_classes, ax=axs[4, 1])
            axs[4, 1].set_title(f'Path History Heatmap ')
            axs[4, 1].set_xlabel('Output Classes')
            axs[4, 1].set_ylabel('Epochs')

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Save the plots to an image file
        plt.savefig(plot_path)


def make_paths(depth):
    paths = []
    n = 0
    for i in range(pow(2, depth)):
        binary = np.binary_repr(i, width=depth)
        actual_path = ''
        for i in range(depth):
            if binary[i] == '1':
                actual_path += 'R'
            else:
                actual_path += 'L'
        paths.append(actual_path)
        n += 1
    return paths
