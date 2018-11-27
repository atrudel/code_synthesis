import os
import time, datetime
import config
import torch
from game.environment import Env
from tensorboardX import SummaryWriter
from collections import deque
import numpy as np

CFG = config.get_cfg()
CWCFG = CFG.settings.toy_corewar

class Agent:
    def __init__(self, verbose, log_dir):
        self.best_score = -float('Inf')
        self.best_episode = 0
        self.verbose = verbose
        self.log_dir = log_dir
        self.writer = SummaryWriter() if log_dir else None
        self.model = None
        self.best_model = None
        self.total_episodes = 0
        self.performances = deque(maxlen=CFG.settings.PERF_MRY)
    
    ## Methods that need to be implemented in the child classes
    
    def train(self, reward_func, reg_init, episodes):
        raise NotImplementedError("You need to implement a train method in your class!")
    
    def act(self, state):
        raise NotImplementedError("You need to implement an act method in your class!")
    
    def load(self, path):
        raise NotImplementedError("You need to implement an act method in your class!")

        
    ## Methods that are implemented in the Agent class

    def save(self, name, best=False):
        if self.log_dir is not None:
            path = os.path.join(self.log_dir, "models")
            os.makedirs(path, exist_ok=True)
            path = os.path.join(path, name)
        else:
            path = name
        if best:
            torch.save(self.best_model.state_dict(), path)
        else:
            torch.save(self.model.state_dict(), path)


    def assess(self, reward_func, reg_init=None, episode=None, print=False, file=None):
        env = Env(reward_func)
        s = env.reset(reg_init)
        for t in range(CWCFG.MAX_LENGTH):
            a = self.act(s)
            s_prime, reward, done, _ = env.step(a)
            s = s_prime
            if done:
                break
        performance = reward_func.performance(env)
        total_reward = env.total_reward
        
        if print:
            env.print_details(file=file)
        return performance, total_reward


    def evaluate(self, log=False):
        performances = []
        total_rewards = []
        if log and self.log_dir:
            filename = os.path.join(self.log_dir, "{:07}_Evaluation".format(self.total_episodes))
        else:
            filename = os.devnull
        with open(filename, 'w') as f:
            print("Evaluation over {} tasks".format(len(self.tasks)), file=f)
            print("Algorithm: {}".format(self.__class__.__name__), file=f)
            current_time = datetime.timedelta(seconds=(time.time() - self.start_time))
            print("Time: {}".format(str(current_time)), file=f)

            for reward_func, reg_init in self.tasks:
                print("Task:  {}".format(reward_func), file=f)
                print("Initialization: {}".format(reg_init), file=f)
                performance, total_reward = self.assess(reward_func, reg_init, print=True, file=f)
                performances.append(performance)
                total_rewards.append(total_reward)
                print("\n\n", file=f)

            mean_perf = np.mean(performances)
            mean_reward = np.mean(total_rewards)
            print("Mean performance: {}".format(mean_perf), file=f)
            print("Mean reward: {}".format(mean_reward), file=f)

        if self.verbose:
            print("Currently at episode {}".format(self.total_episodes))
        if log and self.writer is not None:
            self.writer.add_scalars(self.log_dir, {'Mean performance': mean_perf}, self.total_episodes)
            self.writer.add_scalars(self.log_dir, {'Mean total reward': mean_reward}, self.total_episodes)

        if mean_perf > self.best_score:
            self.best_score = mean_perf
            self.best_episode = self.total_episodes
            self.best_model.load_state_dict(self.model.state_dict())
        self.performances.append(mean_perf)
        return mean_perf, mean_reward


    def generalize(self, Reward_func, num_tasks, reward_settings=None, reg_zero_init=True, log=False):
        np.random.seed(0)
        performances = []
        total_rewards = []
        if log and self.log_dir:
            filename = os.path.join(self.log_dir, "{:07}_Generalization".format(self.total_episodes))
        else:
            filename = os.devnull
        with open(filename, 'w') as f:
            print("Generalization over {} test tasks".format(num_tasks), file=f)
            print("Algorithm: {}".format(self.__class__.__name__), file=f)
            current_time = datetime.timedelta(seconds=(time.time() - self.start_time))
            print("Time: {}".format(str(current_time)), file=f)
            for n in range(num_tasks):
                reward_function = Reward_func(None, reward_settings)
                if reg_zero_init:
                    reg_init = np.zeros(CWCFG.NUM_REGISTERS, dtype=int)
                else:
                    reg_init = np.random.randint(0, 256, CWCFG.NUM_REGISTERS)
                print("Task {}:  {}".format(n + 1, reward_function), file=f)
                print("Initialization: {}".format(reg_init), file=f)
                performance, total_reward = self.assess(reward_function, reg_init, print=True, file=f)
                performances.append(performance)
                total_rewards.append(total_reward)
                print("\n\n", file=f)
            mean_perf = np.mean(performances)
            mean_reward = np.mean(total_rewards)
            print("Mean performance: {}".format(mean_perf), file=f)
            print("Mean reward: {}".format(mean_reward), file=f)
        np.random.seed()

        if self.verbose:
            print("Currently at episode {}".format(self.total_episodes))
        if log and self.writer is not None:
            self.writer.add_scalars(self.log_dir, {'Mean performance': mean_perf}, self.total_episodes)
            self.writer.add_scalars(self.log_dir, {'Mean total reward': mean_reward}, self.total_episodes)

        if mean_perf > self.best_score:
            self.best_score = mean_perf
            self.best_episode = self.total_episodes
            self.best_model.load_state_dict(self.model.state_dict())
        self.performances.append(mean_perf)
        return mean_perf, mean_reward

    def best_performance(self):
        return self.best_score, self.best_episode

    def global_performance(self):
        return np.mean(self.performances)

    def final_performance(self):
        return self.performances[-1]

    def __del__(self):
        if self.writer is not None:
            self.writer.close()
