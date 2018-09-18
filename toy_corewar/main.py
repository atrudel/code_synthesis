import os
from model import Dueling_DQN
from train import train
from reward import *
import torch
import multiprocessing
from multiprocessing import Pool
from itertools import repeat

def run_experiment(reward_func, episodes, root_dir):
    log_dir = os.path.join(root_dir, reward_func.__name__)
    os.makedirs(log_dir)
    Q = Dueling_DQN()
    train(Q, reward_func, episodes, log_dir=log_dir, verbose=True)
    final_save_path = os.path.join(log_dir, "models", "final")
    torch.save(Q.state_dict(), final_save_path)
    
def run_experiment_series(name, reward_functions, episodes):
    os.makedirs("Experiments", exist_ok=True)
    root_dir = os.path.join("Experiments", name)
    os.makedirs(root_dir)
    
    if isinstance(episodes, int):
        episodes = [episodes] * len(reward_functions)  
    
    if MULTIPROCESSING:
        multiprocessing.set_start_method('spawn')
        with Pool(processes=len(reward_functions)) as pool:
            pool.starmap(run_experiment, zip(reward_functions, episodes, repeat(root_dir)))
    else:
        for reward_func, ep in zip(reward_functions, episodes):
            run_experiment(reward_func, ep, root_dir)

            
if __name__ == '__main__':
     run_experiment_series("04-Small_LSTM", reward_functions, 50000)
#    run_experiment_series("test", [maximize_all_registers], 500)
