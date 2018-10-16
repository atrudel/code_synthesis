import os
import inspect
import time, datetime
import config
import torch
from game.environment import Env
from tensorboardX import SummaryWriter

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
    
    
    ## Methods that need to be implemented in the child classes
    
    def train(self, reward_func, episodes):
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
    
    def assess(self, reward_func, episode=None, print=False, file=None):
        env = Env(reward_func)
        s = env.reset()
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
    
    def best_performance(self):
        return self.best_score, self.best_episode
    
    def log_init(self, episodes, reward_func):
        # Console output
        if self.verbose:
            print("Starting training [algo = {}, reward = {}] for {} episodes...".format(
                self.__class__.__name__, reward_func.__name__, episodes))
        # Logging file output
        if self.log_dir is not None:
            with open(os.path.join(self.log_dir, "logs"), "w") as f:
                print("Starting training for {} episodes...".format(episodes), file=f)
                print("Algorithm: {}".format(self.__class__.__name__), file=f)
                print("Reward function:\n\n{}\n\n\n".format(inspect.getsource(reward_func)), file=f)
    
    def log(self, episode, reward_func, start_time):
        # to console
        if self.verbose:
            print("Episode {} completed for {}, {}".format(
                episode + 1, self.__class__.__name__, reward_func.__name__))
        
        # to log file and Tensorboard
        if self.log_dir is not None:
            with open(os.path.join(self.log_dir, "logs"), "a") as f:
                current_time = datetime.timedelta(seconds=(time.time()-start_time))
                print("Episode {}: [time:  {}]\n".format(episode+1, str(current_time)), file=f)
                score = self.assess(reward_func, episode=episode, print=True, file=f)
                print("\n\n\n", file=f)
                # log to Tensorboard
                self.writer.add_scalars(self.log_dir, {'rewards': score}, episode)
    
    def __del__(self):
        if self.writer is not None:
            self.writer.close()
