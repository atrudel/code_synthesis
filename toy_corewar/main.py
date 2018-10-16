import config
config.load("config.json")
cfg = config.get_cfg()

import os
from DQN.DQN_agent import DQN_Agent
from reward import *
import torch
import multiprocessing
from multiprocessing import Pool
from itertools import repeat
import inspect
import argparse

agents = {
    'DQN_Agent': DQN_Agent
}


def run_experiment(preset, reward_func, episodes, root_dir):
    
    # Conduct experiment and output logs in a separate directory
    preset = getattr(cfg.presets, preset)
    log_dir = os.path.join(root_dir, reward_func.__name__)
    os.makedirs(log_dir)
    agent = agents[preset.agent](**vars(preset.parameters), verbose=True, log_dir=log_dir)
    agent.train(reward_func, episodes)
    score, episode = agent.best_performance()
    
    # Output best score and corresponding episode in the summary
    with open(os.path.join(root_dir, "Summary"), "a") as f:
        print("Reward function: \n\n{}\nScore:   {}\nEpisode: {}\n\n".format(
            inspect.getsource(reward_func), score, episode), file=f)
    
    # Save best performing Agent
    agent.save("best", best=True)
    
def run_experiment_series(preset, name, reward_functions, episodes):
    os.makedirs("Experiments", exist_ok=True)
    root_dir = os.path.join("Experiments", name)
    os.makedirs(root_dir)
    
    if isinstance(episodes, int):
        episodes = [episodes] * len(reward_functions)  
    
    if cfg.settings.MULTIPROCESSING:
        multiprocessing.set_start_method('spawn')
        with Pool(processes=len(reward_functions)) as pool:
            pool.starmap(run_experiment, zip(repeat(preset), reward_functions, 
                                                episodes, repeat(root_dir)))
    else:
        for reward_func, ep in zip(reward_functions, episodes):
            run_experiment(preset, reward_func, ep, root_dir)

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run an experiment series')
    parser.add_argument('name', help="name of the experiment")
    parser.add_argument('-e', '--episodes', type=int, default=cfg.settings.MAX_EPISODES,
        help='number of episodes for each training')
    parser.add_argument('-p', '--preset', default='DQN', help='preset to use')
    args = parser.parse_args()
    
    run_experiment_series(args.preset, args.name, [maximize_all_registers], args.episodes)
