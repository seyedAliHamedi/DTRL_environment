import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

from environment.model.actor_critic import ActorCritic
from environment.model.core_scheduler import CoreScheduler


class Agent(mp.Process):
    def __init__(self, name, global_actor_critic, optimizer, barrier, shared_state):
        super(Agent, self).__init__()
        self.global_actor_critic = global_actor_critic
        self.local_actor_critic = ActorCritic(
            self.global_actor_critic.input_dims, self.global_actor_critic.n_actions)
        self.optimizer = optimizer
        self.name = name
        self.num_input = 5
        self.num_output = 5
        self.devices = shared_state.database.get_all_devices()
        self.assigned_job = None
        self.core = CoreScheduler(self.devices)
        self.runner_flag = mp.Value('b', True)
        self.barrier = barrier
        self.state = shared_state

    def init_logs(self):
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
        self.fails_log=0
        # usuage and explore
        self.iot_usuage=0
        self.mec_usuage=0
        self.cc_usuage=0


    def run(self):
        while self.runner_flag:
            self.barrier.wait()
            if self.assigned_job is None:
                self.assigned_job = self.state.preprocessor.assign_job()
                if self.assigned_job is None:
                    continue
                self.local_actor_critic.clear_memory()
                self.init_logs()
            agent_queue=self.state.preprocessor.get_agent_queue()
            task_queue = agent_queue.get(self.assigned_job)
            for task in task_queue:
                self.schedule(task)

            try:
                current_job = self.state.get_job(self.assigned_job)
            except:
                current_job = self.state.get_job(self.assigned_job)
            if current_job and len(current_job["runningTasks"]) + len(current_job["finishedTasks"]) == current_job["task_count"]:
                print("DONE",self.assigned_job)
                self.update()
                self.assigned_job = None

    def stop(self):
        self.runner_flag = False

    def schedule(self, current_task_id):
        job_state, pe_state = self.state.get()
        current_task = self.state.database.get_task(current_task_id)
        current_job = self.state.get_job(self.assigned_job)
        input_state = self.get_input(current_task, {})

        option = self.local_actor_critic.choose_action(input_state)

        selected_device_index = option
        selected_device = self.devices[option]

        if selected_device['type'] != "cloud":
            sub_state = self.get_input(
                current_task, {0: pe_state[selected_device['id']]})
            sub_state = torch.tensor(sub_state, dtype=torch.float32)
            action_logits = self.core.forest[selected_device_index](sub_state)
            action_dist = torch.distributions.Categorical(
                F.softmax(action_logits, dim=-1))
            action = action_dist.sample()
            selected_core_index = action.item()
            selected_core = selected_device["voltages_frequencies"][selected_core_index]
            dvfs = selected_core[np.random.randint(0, 3)]

        if selected_device['type'] == "cloud":
            selected_core_index = -1
            i = np.random.randint(0, 1)
            dvfs = [(50000, 13.85), (80000, 24.28)][i]
        reward, fail_flag, energy, time = self.state.apply_action(
            selected_device_index, selected_core_index, dvfs[0], dvfs[1], current_task_id)
        
        self.local_actor_critic.archive(input_state, option, reward)
        self.reward_log.append(reward)
        self.time_log.append(time)
        self.energy_log.append(energy)
        if fail_flag[0]:
            self.safe_fails_log +=1
        if fail_flag[1]:
            self.kind_fails_log +=1
        if fail_flag[2]:
            self.queue_fails_log +=1
        self.fails_log+= sum(fail_flag)
        if selected_device['type']=="iot":
            self.iot_usuage+=1
        if selected_device['type']=="mec":
            self.mec_usuage+=1
        if selected_device['type']=="cloud":
            self.cc_usuage+=1
    def update(self):

        loss = self.local_actor_critic.calc_loss()
        with self.state.lock:
            self.state.agent_log[self.assigned_job] = {
                "loss": loss.item(),
                "reward": sum(self.reward_log)/len(self.reward_log),
                "time": sum(self.time_log)/len(self.time_log),
                "energy": sum(self.energy_log)/len(self.energy_log),
                "safe_fails": self.safe_fails_log/len(self.energy_log),
                "kind_fails": self.kind_fails_log/len(self.energy_log),
                "queue_fails": self.queue_fails_log/len(self.energy_log),
                "fails": self.fails_log/len(self.energy_log),
                "iot_usuage": self.iot_usuage/len(self.energy_log),
                "mec_usuage": self.mec_usuage/len(self.energy_log),
                "cc_usuage": self.cc_usuage/len(self.energy_log),
            }
        loss.backward()
        for local_param, global_param in zip(self.local_actor_critic.parameters(),
                                             self.global_actor_critic.parameters()):
            global_param._grad = local_param.grad
        self.optimizer.step()
        self.local_actor_critic.load_state_dict(
            self.global_actor_critic.state_dict())
        self.local_actor_critic.clear_memory()

        return loss


####### UTILITY #######

    def get_pe_data(self, pe_dict):

        pe = self.state.database.get_device(pe_dict["id"])
        battery_capacity = pe["battery_capacity"]
        battery_level = pe_dict["batteryLevel"]
        battery_isl = pe["ISL"]
        battery = (battery_level / battery_capacity -
                   battery_isl) * battery_capacity

        num_cores = pe["num_cores"]
        cores_availability = pe_dict["occupiedCores"]
        cores = 1 - (sum(cores_availability) / num_cores)

        devicePower = 0
        for index, core in enumerate(pe["voltages_frequencies"]):
            if cores_availability[index] == 1:
                continue
            corePower = 0
            for mod in core:
                freq, vol = mod
                corePower += freq / vol
            devicePower += corePower
        devicePower = devicePower / num_cores

        error_rate = pe["error_rate"]

        return [cores, devicePower, battery, error_rate]

    def get_input(self, task, pe_dict):
        task_features = self.get_task_data(task)
        pe_features = []
        for pe in pe_dict.values():
            pe_features.extend(self.get_pe_data(pe))
        return task_features + pe_features

    def get_task_data(self, task):
        return [
            task["computational_load"],
            task["input_size"],
            task["output_size"],
            task["task_kind"],
            task["is_safe"],
        ]
