import os
from model import Dueling_DQN
from train import *
from reward import *
import torch
import multiprocessing
from multiprocessing import Pool
from itertools import repeat
import inspect


parameters = dict(
    h_size = 35,
    middle_size = 128,
    lstm_layers = 2,
    learning_starts = 100,
    learning_freq = 4,
    target_update_freq = 1000, 
    lr = 0.01,
    gamma = 0.99,
    batch_size = 32,
    replay_buffer_size = 100000
)

def run_experiment(reward_func, episodes, root_dir):
    # Conduct experiment and output logs in a separate directory
    log_dir = os.path.join(root_dir, reward_func.__name__)
    os.makedirs(log_dir)
    DDQN, score, episode = train_DQN(reward_func, episodes, **parameters, epsilon_decay_steps=episodes, log_dir=log_dir, verbose=True)
    
    # Output best score and corresponding episode in the summary
    with open(os.path.join(root_dir, "Summary"), "a") as f:
        print("Reward function: \n\n{}\nScore:   {}\nEpisode: {}\n\n".format(
            inspect.getsource(reward_func), score, episode), file=f)
    
    # Save best performing DDQN
    best_save_path = os.path.join(log_dir, "models", "best")
    torch.save(DDQN.state_dict(), best_save_path)
    
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
    run_experiment_series("bibi", reward_functions, 500)
#   run_experiment_series("test_LSTM_fusion", [maximize_all_registers], 50000)
