import os
import time, datetime
import config
import torch
from game.environment import Env
from tensorboardX import SummaryWriter
from collections import namedtuple
import numpy as np

CFG = config.get_cfg()
CWCFG = CFG.settings.toy_corewar

Task = namedtuple('Task', ('reward_function', 'reg_init', 'total_episodes', 'best_score'))

class Agent:
    def __init__(self, verbose, log_dir):
        self.best_score = -float('Inf')
        self.best_episode = 0
        self.verbose = verbose
        self.log_dir = log_dir
        self.log_num = 0
        self.writer = SummaryWriter() if log_dir else None
        self.model = None
        self.best_model = None
        self.total_episodes = 0
    
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
        score = env.total_reward
        
        # If an episode number was specified, the score can break the agent's record
        if score > self.best_score and episode is not None:
            self.best_score = score
            self.best_episode = episode + 1
            self.best_model.load_state_dict(self.model.state_dict())
        
        if print:
            env.print_details(file=file)
        return score

    def generalize(self, Reward_func, num_tasks, reg_zero_init=True, log=False):
        np.random.seed(0)
        results = []
        if log:
            filename = os.path.join(self.log_dir, "{}_Generalization".format(self.total_episodes))
        else:
            filename = os.devnull
        with open(filename, 'w') as f:
            print("Generalization over {} test tasks".format(num_tasks), file=f)
            print("Algorithm: {}".format(self.__class__.__name__), file=f)
            current_time = datetime.timedelta(seconds=(time.time() - self.start_time))
            print("Time: {}".format(str(current_time)), file=f)
            for n in range(num_tasks):
                reward_function = Reward_func(None)
                if reg_zero_init:
                    reg_init = np.zeros(CWCFG.NUM_REGISTERS, dtype=int)
                else:
                    reg_init = np.random.randint(0, 256, CWCFG.NUM_REGISTERS)
                print("Task:  {}".format(reward_function), file=f)
                print("Initialization: {}".format(reg_init), file=f)
                results.append(self.assess(reward_function, reg_init, print=True, file=f))
                print("\n\n", file=f)
            mean = np.mean(results)
            print("Mean score: {}".format(mean), file=f)
        np.random.seed()

        if mean > self.best_score:
            self.best_score = mean
            self.best_episode = self.total_episodes
            self.best_model.load_state_dict(self.model.state_dict())

        if self.verbose:
            print("Currently at episode {}".format(self.total_episodes))
        self.writer.add_scalars(self.log_dir, {'mean_rewards': mean}, self.total_episodes)
        return mean, results


    def best_performance(self):
        return self.best_score, self.best_episode

    def __del__(self):
        if self.writer is not None:
            self.writer.close()

    # def evaluate(self):
    #     results = []
    #     filename = os.path.join(self.log_dir, "Evaluation")
    #     with open(filename, 'w') as f:
    #         print("Evaluation over {} tasks".format(len(self.tasks)), file=f)
    #         print("Algorithm: {}".format(self.__class__.__name__), file=f)
    #         for task in self.tasks:
    #             print("Task:  {}".format(task.reward_function), file=f)
    #             print("Initialization: {}".format(task.reg_init), file=f)
    #             results.append(self.assess(task.reward_function, task.reg_init, print=True, file=f))
    #             print("\n\n", file=f)
    #         mean = np.mean(results)
    #         print("Mean score: {}".format(mean), file=f)
    #     return mean, results

    # def log_init(self, episodes, reward_func):
    #     self.log_num += 1
    #     # Console output
    #     if self.verbose:
    #         print("Starting training [algo = {}, reward = {}, version {}] for {} episodes...".format(
    #             self.__class__.__name__, reward_func.__class__.__name__, self.log_num, episodes))
    #     # Logging file output
    #     if self.log_dir is not None:
    #         with open(os.path.join(self.log_dir, "logs{}".format(self.log_num)), "w") as f:
    #             print("Starting training for {} episodes...".format(episodes), file=f)
    #             print("Algorithm: {}".format(self.__class__.__name__), file=f)
    #             print("Reward function:  {}\n\n\n".format(reward_func), file=f)
    #

    # def log(self, episode, reward_func, start_time):
    #     # to console
    #     if self.verbose:
    #         print("Episode {} completed for {}, {}_{}".format(
    #             episode + 1, self.__class__.__name__, reward_func.__class__.__name__, self.log_num))
    #
    #     # to log file and Tensorboard
    #     if self.log_dir is not None:
    #         with open(os.path.join(self.log_dir, "logs{}".format(self.log_num)), "a") as f:
    #             current_time = datetime.timedelta(seconds=(time.time()-start_time))
    #             print("Episode {}: [time:  {}]\n".format(episode+1, str(current_time)), file=f)
    #             score = self.assess(reward_func, episode=episode, print=True, file=f)
    #             print("\n\n\n", file=f)
    #             # log to Tensorboard
    #             self.writer.add_scalars(self.log_dir, {'rewards': score}, episode)

# def multi_train(self, Reward_func, targets, reg_init_freq, episodes):
    #     ''' Performs multiple trainings on the same task, but with different target values and register initializations.
    #     Arg types:
    #     - reward_func: [string] name of a Reward_function class
    #     - targets: [integer] number of random target values to generate
    #     - reg_init_freq: [integer] indicates after how many episodes register initializations are randomly reset.
    #                     A value of O means that all registers are initialized at 0.
    #     - episodes: [integer] the total number of episodes done, divided among all training subtasks'''
    #
    #     if isinstance(targets, list) and isinstance(targets[0], np.ndarray):
    #         num_targets = len(targets)
    #     elif episodes % num_targets == 0:
    #         num_targets = targets
    #         episodes_per_target = episodes // num_targets
    #         targets = [None for _ in range(num_targets)] # 'None' means the Reward_function object is constructed with random target values
    #     else:
    #         raise ValueError("Need episodes({}) % num_targets({}) == 0".format(episodes, num_targets))
    #
    #     if reg_init_freq < 0 :
    #         raise ValueError("Negative reg_init_freq: {}".format(reg_init_freq))
    #     elif reg_init_freq == 0:
    #         zero_init = True
    #         reg_init = np.zeros(CWCFG.NUM_REGISTERS, dtype=int)
    #         reg_init_freq = episodes_per_target
    #     elif episodes_per_target % reg_init_freq == 0:
    #         zero_init = False
    #     else:
    #         raise ValueError("Need (episodes({}) // num_targets({})) % reg_init_freq({}) == 0".format(episodes, num_targets, reg_init_freq))
    #
    #     # Create training tasks
    #     for target in targets:
    #         reward = reward_func(target)
    #         for _ in range(episodes_per_target // reg_init_freq):
    #             if not zero_init:
    #                 reg_init = np.random.randint(0, 256, CWCFG.NUM_REGISTERS)
    #             self.tasks.append(Task(reward, reg_init, 0, -float('Inf')))
    #
    #     task = self.tasks[0]
    #     self.train(reward, reg_init, reg_init_freq)
    #     self.save("End_multi_training", best=False)