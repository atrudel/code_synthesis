import config
import numpy as np

CFG = config.get_cfg()
CWCFG = CFG.settings.toy_corewar

class Task_Manager(object):
    def __init__(self, Reward_func, reward_settings, targets, reg_init, episodes):
        self.Reward_func = Reward_func
        self.reward_settings = reward_settings
        self.episodes = episodes

        # Handle target values
        if targets is None:
            self.tasks = [None]
            self.target_altern_freq = 0
        else:
            self.target_altern_freq = targets['alternate_freq']
            if isinstance(targets['values'], int):
                self.tasks = [Reward_func(None, reward_settings)
                              for _ in range(targets['values'])]
            elif isinstance(targets['values'], list):
                self.tasks = [Reward_func(np.array(target, dtype=int), reward_settings)
                              for target in targets['values']]
            else:
                raise ValueError("Unrecognized data type for 'targets': {}".format(targets))

        # Handle register initialization
        if reg_init is None:
            self.reg_inits = [np.zeros(CWCFG.NUM_REGISTERS, dtype=int)]
            self.init_altern_freq = 0
        else:
            self.init_altern_freq = reg_init['alternate_freq']
            if isinstance(reg_init['values'], int):
                self.reg_inits = [np.random.randint(0, 256, CWCFG.NUM_REGISTERS) for _ in range(reg_init['values'])]
            elif isinstance(reg_init['values'], list):
                self.reg_inits = [np.array(reg, dtype=int) for reg in reg_init['values']]
            else:
                raise ValueError("Unrecognized data type for 'reg_init': {}".format(reg_init))

        # Reinterpret alternating frequencies of 0
        if self.target_altern_freq == 0:
            self.target_altern_freq = episodes // len(self.tasks)
        if self.init_altern_freq == 0:
            self.init_altern_freq = episodes // len(self.tasks) // len(self.reg_inits)

    def get_current(self, episode):
        reward_func = self.tasks[episode // self.target_altern_freq % len(self.tasks)]
        if reward_func is None:
            reward_func = self.Reward_func(None, self.reward_settings)

        reg_init = self.reg_inits[episode // self.target_altern_freq % len(self.reg_inits)]

        return reward_func, reg_init

    # This function will cause an error if targets were initialized as None!
    def __iter__(self):
        for reward_func in self.tasks:
            for reg_init in self.reg_inits:
                yield reward_func, reg_init

    def __len__(self):
        return len(self.tasks)