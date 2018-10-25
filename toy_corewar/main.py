import config
config.load("config.json")
cfg = config.get_cfg()

import os, shutil
import reward
import multiprocessing
from multiprocessing import Pool
import argparse
import json
from DQN.DQN_agent import DQN_Agent

def unpack(args):
    run_training(**args)

def run_training(id, algo, episodes, reward_func, reward_settings, targets, reg_init, root_dir):
    log_dir = os.path.join(root_dir, str(id))
    os.makedirs(log_dir)

    preset = getattr(cfg.presets, algo)
    agent = globals()[preset.agent](**preset.parameters.todict(), verbose=True, log_dir=log_dir)
    Reward_func = getattr(reward, reward_func)
    agent.train(Reward_func, reward_settings, episodes, targets, reg_init)
    agent.save("best", best=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a multi-training series')
    parser.add_argument('name', help="name of the experiment")
    parser.add_argument('-f', '--force', help="force overwriting of the log folder", action='store_true')
    parser.add_argument('-t', '--training', help="specify custom training file", default="training.json")
    parser.add_argument('--nomultiprocessing', help="run the program using a single process", action='store_true')
    args = parser.parse_args()

    with open(args.training) as f:
        trainings = json.load(f)

    os.makedirs("Experiments", exist_ok=True)

    root_dir = os.path.join("Experiments", args.name)
    if args.force:
        shutil.rmtree(root_dir, ignore_errors=True)
    os.makedirs(root_dir)
    with open(os.path.join(root_dir, "Training_descriptions.json"), 'w') as f:
        json.dump(trainings, f, indent=4)
        # json.dump(cfg.todict(), f, indent=4) doesn't work

    for training in trainings:
        training['root_dir'] = root_dir

    if not args.nomultiprocessing:
        multiprocessing.set_start_method('spawn')
        with Pool(processes=multiprocessing.cpu_count()) as pool:
            pool.map(unpack, trainings)
            # pool.starmap(run_multi_training, trainings)
    else:
        for training in trainings:
            unpack(training)
