import os
from model import Dueling_DQN
from train import train
from reward import *
import torch

def run_experiment(reward_func, episodes, root_dir):
    log_dir = os.path.join(root_dir, reward_func.__name__)
    os.makedirs(log_dir)
    Q = Dueling_DQN()
    train(Q, reward_func, episodes, log_dir=log_dir)
    final_save_path = os.path.join(log_dir, "models", "final")
    torch.save(Q.state_dict(), final_save_path)
    
def run_experiment_series(name, reward_functions, episodes):
    os.makedirs("Experiments", exist_ok=True)
    root_dir = os.path.join("Experiments", name)
    os.makedirs(root_dir)
    if isinstance(episodes, int):
        episodes = [episodes] * len(reward_functions)
    for reward_func, ep in zip(reward_functions, episodes):
        run_experiment(reward_func, ep, root_dir)

run_experiment_series("02-No-cummulative_Reward", reward_functions, 50000)
