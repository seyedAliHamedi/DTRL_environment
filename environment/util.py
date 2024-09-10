import numpy as np


def reward_function(setup=5, e=0, alpha=1, t=0, beta=1, punish=0):
    if punish:
        return -10

    if setup == 1:
        return -1 * (alpha * e + beta * t)
    elif setup == 2:
        return 1 / (alpha * e + beta * t)
    elif setup == 3:
        return -np.exp(alpha * e) - np.exp(beta * t)
    elif setup == 4:
        return -np.exp(alpha * e + beta * t)
    elif setup == 5:
        return np.exp(-1 * (alpha * e + beta * t))
    elif setup == 6:
        return -np.log(alpha * e + beta * t)
    elif setup == 7:
        return -((alpha * e + beta * t) ** 2)


def find_place(pe, core_i):
    for i, slot in enumerate(pe["queue"][core_i]):
        if slot[1] == -1:
            return i, core_i
    return -1, -1


def check_fail(pe, queue_index, core_index, task):
    fail_flag = [0, 0, 0, 0]
    if task["is_safe"] and not pe['handleSafeTask']:
        # fail : assigned safe task to unsafe device
        fail_flag[0] = 0
    elif task["task_kind"] not in pe["acceptableTasks"]:
        # fail : assigned a kind of task to the inappropriate device
        fail_flag[1] = 0
    elif queue_index == -1 and core_index == -1:
        # fail : assigned a task to a full queue core
        fail_flag[2] = 1
    return fail_flag
