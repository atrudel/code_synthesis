import config
config.load("config.json")
cfg = config.get_cfg()

import os, shutil
from DQN.DQN_agent import DQN_Agent
import reward
import torch
import multiprocessing
from multiprocessing import Pool
from itertools import repeat
import inspect
import argparse
import json

trainings = [
    {
        'id': 1,
        'algo': 'DQN',
        'reward_func': "Specific_register_values",
        'episodes': 50000,
        'num_targets': 5,
        'reg_init_freq': 0,
        'mode': 'train',
        'threshold': 0
    },
    {
        'id': 2,
        'algo': 'DQN',
        'reward_func': "Specific_register_values",
        'episodes': 50000,
        'num_targets': 50000,
        'reg_init_freq': 1,
        'mode': 'train',
        'threshold': 0
    },
    {
        'id': 3,
        'algo': 'DQN',
        'reward_func': "One_register_value",
        'episodes': 50000,
        'num_targets': 500,
        'reg_init_freq': 0,
        'mode': 'train',
        'threshold': 0
    }
]

def unpack(args):
    run_multi_training(**args)

def run_multi_training(id, algo, reward_func, episodes, num_targets, reg_init_freq, mode, threshold, root_dir):
    log_dir = os.path.join(root_dir, str(id))
    os.makedirs(log_dir)

    preset = getattr(cfg.presets, algo)
    agent = globals()[preset.agent](**preset.parameters.todict(), verbose=True, log_dir=log_dir)
    reward_func = getattr(reward, reward_func)
    agent.multi_train(reward_func, num_targets, reg_init_freq, episodes)

    score, episode = agent.best_performance()

    # Output best score and corresponding episode in the summary
    with open(os.path.join(root_dir, "Summary"), "a") as f:
        print("Training {}", file=f)

    # Save best performing Agent
    agent.save("best", best=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a multi-training series')
    parser.add_argument('name', help="name of the experiment")
    parser.add_argument('-f', '--force', help="force overwriting of the log folder", action='store_true')
    args = parser.parse_args()

    os.makedirs("Experiments", exist_ok=True)
    root_dir = os.path.join("Experiments", args.name)
    if args.force:
        shutil.rmtree(root_dir, ignore_errors=True)
    os.makedirs(root_dir)
    with open(os.path.join(root_dir, "Training_descriptions.json"), 'w') as f:
        json.dump(trainings, f, indent=4)
        # TODO: dump configuration file also

    for training in trainings:
        training['root_dir'] = root_dir


    if cfg.settings.MULTIPROCESSING:
        multiprocessing.set_start_method('spawn')
        with Pool(processes=len(trainings)) as pool:
            pool.map(unpack, trainings)
            # pool.starmap(run_multi_training, trainings)
    else:
        for training in trainings:
            unpack(training)
