import torch
from collections import deque, namedtuple
from config import *
from game.environment import Env

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

class LinearSchedule(object):
    def __init__(self, schedule_episodes, total_episodes, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_episodes. After this many episodes pass final_p is
        returned.
        Parameters
        ----------
        schedule_episodes: int
            Number of episodes for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        if isinstance(schedule_episodes, float):
            self.schedule_episodes = int(schedule_episodes * total_episodes)
        else:
            self.schedule_episodes = schedule_episodes
        self.final_p            = final_p
        self.initial_p          = initial_p

    def value(self, t):
        fraction  = min(float(t) / self.schedule_episodes, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)

def state_to_tensors(state):
    prog_state, mem_state = state
    prog = torch.tensor(prog_state).unsqueeze(1).to(DEVICE)
    mem = torch.tensor(mem_state).view(-1).unsqueeze(0).to(DEVICE)
    return prog, mem

def batch_to_tensors(batch):
    tensors = [state_to_tensors(state) for state in batch]
    prog_tensors, mem_tensors = zip(*tensors)
    prog = torch.cat(prog_tensors, dim=1)
    mem = torch.cat(mem_tensors, dim=0)
    return prog, mem
