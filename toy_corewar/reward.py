import config
import numpy as np
from random import randint as randint
from game.toyCorewar import ToyCorewar

CFG = config.get_cfg()
CWCFG = CFG.settings.toy_corewar


class Reward_function(object):
    '''targets is a numpy array of shape=(4,), dtype=int
        settings is a dictionary defined in config.json'''
    def __init__(self, targets=None):
        self.targets = targets if targets is not None else self.random()

    def __call__(self, env):
        return self.reward(env)

    def reward(self, env):
        pass

    def performance(self, env):
        pass

    '''Generates a random set of target values, adapted for the reward function'''
    def random(self):
        pass

    def __str__(self):
        return "({}: target={})".format(self.__class__.__name__, str(self.targets))


# Specific values for all registers
class Specific_register_values(Reward_function):

    def __init__ (self, targets=None, settings=None):
        super(Specific_register_values, self).__init__(targets)

        # Make it compatible to non-defined settings (None)
        if settings['circular']:
            self.distance = self.circular_distance
        else:
            self.distance = self.absolute_distance

        if settings['positive']:
            self.reward = self.positive_reward
            self.maximum = 128 * CWCFG.N_TARGETS
        else:
            self.reward = self.negative_reward
            self.maximum = 0
        self.circular = settings['circular']
        self.positive = settings['positive']
        self.cumulative = settings['cumulative']

    def positive_reward(self, env):
        initial_dist = self.distance(env.register_states[-2], self.targets)
        final_dist = self.distance(env.register_states[-1], self.targets)
        improvement = initial_dist - final_dist
        return improvement.sum()

    def negative_reward(self, env):
        final_dist = self.distance(env.register_states[-1], self.targets)
        if self.cumulative or env.done:
            return -final_dist.sum()
        else:
            return 0

    def performance(self, env):
        dist = self.distance(env.register_states[-1], self.targets)
        return -dist.sum()

    def absolute_distance(self, before, after):
        diff = before - after
        return abs(diff)

    def circular_distance(self, before, after):
        max = np.maximum(before, after)
        min = np.minimum(before, after)
        diff = max - min
        circ = 256 - max + min
        return np.minimum(diff, circ)

    def random(self):
        return np.random.randint(0, 256, CWCFG.N_TARGETS)
