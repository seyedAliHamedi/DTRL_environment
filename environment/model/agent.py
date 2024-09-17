import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from environment.model.actor_critic import ActorCritic, CoreScheduler


class Agent(mp.Process):
    def __init__(self, name, global_actor_critic, global_optimizer, barrier, shared_state, time_out_counter, max_fail):
        super(Agent, self).__init__()
        # the worker/agent name
        self.name = name
        # the database devices
        self.devices = shared_state.database.get_all_devices()

        # the global Actor-Critic and the gloval optimzer(Shared Adam)
        self.global_actor_critic = global_actor_critic
        self.global_optimizer = global_optimizer
        # the local actor-critic and the core scheduler
        self.local_actor_critic = ActorCritic(self.global_actor_critic.input_dims, self.global_actor_critic.n_actions,
                                              devices=self.devices)
        self.core = CoreScheduler(self.devices)

        # the current assigned job to the agent
        self.assigned_job = None
        self.current_fail_counter = 0
        self.max_fail = max_fail

        # the runner flag and the barrier for the workers process
        self.runner_flag = mp.Value('b', True)
        self.barrier = barrier

        # the shared state
        self.state = shared_state

        self.time_out_counter = time_out_counter
        self.t_counter = 0

    def init_logs(self):
        # intilizing the agent_log
        self.state.agent_log[self.assigned_job] = {}
        self.reward_log = []
        # reward based objectives
        self.time_log = []
        self.energy_log = []
        # punish based objectives
        self.kind_fails_log = 0
        self.safe_fails_log = 0
        self.queue_fails_log = 0
        self.battery_fails_log = 0
        self.fails_log = 0
        # usuage and explore
        self.iot_usuage = 0
        self.mec_usuage = 0
        self.cc_usuage = 0

        self.path_history = []

    def exp_value(self):
        return self.local_actor_critic.exp.value

    def run(self):
        while self.runner_flag:
            self.barrier.wait()
            if self.assigned_job is None:
                self.assigned_job = self.state.assign_job_to_agent()
                if self.assigned_job is None:
                    self.t_counter += 1
                    if self.t_counter >= self.time_out_counter:
                        print(f'Agent {self.name}  TIMEOUT ( no job to be assigned) !!!')
                    continue
                self.local_actor_critic.clear_memory()
                self.init_logs()

            task_queue = self.state.preprocessor.get_agent_queue().get(self.assigned_job)
            if task_queue is None:
                continue
            self.t_counter += 1

            for task in task_queue:
                fails = self.schedule(task)
                self.current_fail_counter += fails
            if self.current_fail_counter >= self.max_fail or self.t_counter >= self.time_out_counter:
                self.skip_job()
            try:
                current_job = self.state.get_job(self.assigned_job)
            except:
                pass
            if current_job and len(current_job["runningTasks"]) + len(current_job["finishedTasks"]) == current_job[
                "task_count"]:
                self.t_counter = 0
                self.current_fail_counter = 0
                print("DONE")
                self.update()
                self.assigned_job = None

    def skip_job(self):
        print(f'NDONE')
        self.state.remove_job(self.assigned_job)
        # TODO drop job if bugged, need to bugfix
        self.update()
        self.assigned_job = None
        self.t_counter = 0
        self.current_fail_counter = 0

    def stop(self):
        self.runner_flag = False

    def schedule(self, current_task_id):
        # Multiprocess Robustness
        try:
            # retrieve the necessary data
            job_state, pe_state = self.state.get()
            current_task = self.state.database.get_task_norm(current_task_id)
            # TODO for feeding tree features give pe_dict correctly
            input_state = self.get_input(current_task, {})
        except:
            print("Retrying schedule on : ", self.name)
            self.schedule(current_task_id)

        # first-level schedule , select a device
        option, path, devices, pis = self.local_actor_critic.choose_action(input_state)
        selected_device = devices[option]
        selected_device_index = self.devices.index(selected_device)
        # second-level schedule for non cloud PEs , select a core and a Voltage Frequency Pair
        sub_state = self.get_input(current_task, {0: pe_state[selected_device['id']]})
        sub_state = torch.tensor(sub_state, dtype=torch.float32)
        action_logits = self.core.forest[selected_device_index](sub_state)
        action_dist = torch.distributions.Categorical(F.softmax(action_logits, dim=-1))
        action = action_dist.sample()
        selected_core_dvfs_index = action.item()
        selected_core_index = int(selected_core_dvfs_index / 3)
        selected_core = selected_device["voltages_frequencies"][selected_core_index]
        dvfs = selected_core[selected_core_dvfs_index % 3]

        # applying action on the state and retrieving the result
        reward, fail_flag, energy, time = self.state.apply_action(
            selected_device_index, selected_core_index, dvfs[0], dvfs[1], current_task_id)

        # print(f'reward for e:{energy},t:{time} -> {reward}|'
        #       f'task_cl:{self.state.database.get_task(current_task_id)["computational_load"]} '
        #       f'with dev{self.state.database.get_device(selected_device_index)["type"]}')

        if action:
            sub_tree_loss = (-action_dist.log_prob(action) * reward)
            self.core.optimizers[selected_device_index].zero_grad()
            sub_tree_loss.backward()
            self.core.optimizers[selected_device_index].step()

        # archive the result to the agent 
        self.local_actor_critic.archive(input_state, option, reward, pis)

        # saving agent logs 
        self.update_agent_logs(reward, time, energy, fail_flag, selected_device, path)

        return sum(fail_flag)

    def update_agent_logs(self, reward, time, energy, fail_flag, selected_device, path):
        self.reward_log.append(reward)
        self.time_log.append(time)
        self.energy_log.append(energy)
        if fail_flag[0]:
            self.safe_fails_log += 1
        if fail_flag[1]:
            self.kind_fails_log += 1
        if fail_flag[2]:
            self.queue_fails_log += 1
        if fail_flag[3]:
            self.battery_fails_log += 1
        self.fails_log += sum(fail_flag)
        if selected_device['type'] == "iot":
            self.iot_usuage += 1
        if selected_device['type'] == "mec":
            self.mec_usuage += 1
        if selected_device['type'] == "cloud":
            self.cc_usuage += 1
        self.fails_log += sum(fail_flag)
        self.path_history.append(path)

    def save_agent_log(self, loss):
        result = {
            "loss": loss,
            "reward": sum(self.reward_log) / len(self.reward_log),
            "time": sum(self.time_log) / len(self.time_log),
            "energy": sum(self.energy_log) / len(self.energy_log),
            "safe_fails": self.safe_fails_log / len(self.energy_log),
            "kind_fails": self.kind_fails_log / len(self.energy_log),
            "queue_fails": self.queue_fails_log / len(self.energy_log),
            "battery_fails": self.battery_fails_log / len(self.energy_log),
            "fails": self.fails_log / len(self.energy_log),
            "iot_usuage": self.iot_usuage / len(self.energy_log),
            "mec_usuage": self.mec_usuage / len(self.energy_log),
            "cc_usuage": self.cc_usuage / len(self.energy_log),
        }
        self.state.save_agent_log(self.assigned_job, result, self.path_history)

    def update(self):
        # updating the agent parameters
        # calculating the loss
        loss = self.local_actor_critic.calc_loss()

        # sending back the result to the state
        self.save_agent_log(loss.item())

        # update params 
        loss.backward()

        # set global params and load them again
        for local_param, global_param in zip(self.local_actor_critic.parameters(),
                                             self.global_actor_critic.parameters()):
            global_param._grad = local_param.grad
        self.global_optimizer.step()
        self.local_actor_critic.load_state_dict(self.global_actor_critic.state_dict())

    ####### UTILITY #######

    def get_input(self, task, pe_dict):
        task_features = self.get_task_data(task)
        pe_features = []
        for pe in pe_dict.values():
            pe_features.extend(self.get_pe_data(pe, pe['id']))
        return task_features + pe_features

    def get_pe_data(self, pe_dict, pe_id):
        pe = self.state.database.get_device(pe_id)
        devicePower = pe['devicePower']

        batteryLevel = pe_dict['batteryLevel']
        battery_capacity = pe['battery_capacity']
        battery_isl = pe['ISL']
        battery = ((1 - battery_isl) * battery_capacity - batteryLevel) / battery_capacity

        num_cores = pe['num_cores']
        cores = 1 - (sum(pe_dict['occupiedCores']) / num_cores)

        return [cores, devicePower, battery]

    def get_task_data(self, task):
        return [
            task["computational_load"],
            task["input_size"],
            task["output_size"],
            task["kind1"],
            task["kind2"],
            task["kind3"],
            task["kind4"],
            task["is_safe"],
        ]
