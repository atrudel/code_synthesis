import torch
from collections import deque, namedtuple
from config import *
from environment import Env

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p            = final_p
        self.initial_p          = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction  = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)

def state_to_tensors(state):
    prog_state, mem_state = state
    instr, var, val = zip(*prog_state)
    instr = torch.tensor(instr).unsqueeze(1)
    var = torch.tensor(var).unsqueeze(1)
    val = torch.tensor(val).unsqueeze(1)
    mem = torch.tensor(mem_state).view(-1).unsqueeze(0)
    return instr, var, val, mem

def batch_to_tensors(batch):
    tensors = [state_to_tensors(state) for state in batch]
    instr_tensors, var_tensors, val_tensors, mem_tensors = zip(*tensors)
    instr = torch.cat(instr_tensors, dim=1)
    var = torch.cat(var_tensors, dim=1)
    val = torch.cat(val_tensors, dim=1)
    mem = torch.cat(mem_tensors, dim=0)
    return instr, var, val, mem

def assess(Q, reward_func, file=None):
    env = Env(reward_func)
    s = env.reset()
    
    for t in range(MAX_LENGTH):
        a = Q(state_to_tensors(s)).argmax(1).item()
        s_prime, reward, done, info = env.step(a)
        s = s_prime
        if done:
            break
    env.print_details(file=file)
    return env