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


class Exploration:
    def __init__(self, starting_exp, decay, min_exp):
        self.value = starting_exp
        self._decay = decay
        self._min_exp = min_exp

    def decay(self):
        if self.value >= self._min_exp:
            self.value -= self._decay
