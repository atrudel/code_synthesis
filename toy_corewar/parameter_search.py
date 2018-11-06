import config
config.load("config.json")
cfg = config.get_cfg()
import reward
from DQN.DQN_agent import DQN_Agent
import os, shutil
import argparse
import json
import sigopt
import math


training = {
        "id": 0,
        "algo": "DQN",
        "episodes": 100000,
        "reward_func": "Specific_register_values",
        "reward_settings": {
            "circular": True,
            "positive": False,
            "cumulative": True
        },
        "targets": {
            "values": [
                [0, 10, 30, 246]
            ],
            "alternate_freq": 0
        },
        "reg_inits": None
}

def run_environment(
    h_size,
    middle_size,
    lstm_layers,
    learning_starts,
    learning_freq,
    target_update_freq,
    lr,
    gamma,
    batch_size,
    replay_buffer_size,
    epsilon_decay_steps,
    final_epsilon,
    root_dir,
    num

):
    log_dir = os.path.join(root_dir, "{:03}".format(num))
    os.makedirs(log_dir)
    agent = DQN_Agent(
        h_size,
        middle_size,
        lstm_layers,
        learning_starts,
        learning_freq,
        target_update_freq,
        lr,
        gamma,
        batch_size,
        replay_buffer_size,
        epsilon_decay_steps,
        final_epsilon,
        verbose=True, log_dir=log_dir)

    Reward_func = getattr(reward, training['reward_func'])
    agent.train(Reward_func, training['reward_settings'], training['episodes'], training['targets'],
                training['reg_inits'])
    agent.save("best", best=True)
    performance = agent.global_performance()
    best_performance, best_episode = agent.best_performance()
    return performance + (1 / (1 + best_episode))

def search(name, root_dir):
    conn = sigopt.Connection(client_token='OQDDPYDWYVUXNFQKIPVCAJRDBWCCHYRARCIBVEHJRJUNOQJQ')

    DQN_experiment = conn.experiments().create(
        name=name,
        observation_budget=300,
        parameters=[
            dict(name='h_size', type='int', bounds=dict(min=20, max=100)),
            dict(name='middle_size', type='int', bounds=dict(min=30, max=250)),
            dict(name='lstm_layers', type='int', bounds=dict(min=1, max=2)),
            dict(name='epsilon_decay_steps', type='double', bounds=dict(min=0.2, max=2.0)),
            dict(name='learning_starts', type='int', bounds=dict(min=0, max=training['episodes'] / 2)),
            dict(name='learning_freq', type='int', bounds=dict(min=1, max=15)),
            dict(name='target_update_freq', type='int', bounds=dict(min=500, max=5000)),
            dict(name='log_lr', type='double', bounds=dict(min=math.log(0.00001), max=math.log(1.0))),
            dict(name='gamma', type='double', bounds=dict(min=0.5, max=0.9999)),
            dict(name='batch_size', type='int', bounds=dict(min=10, max=500))
        ]
    )

    experiment = DQN_experiment
    for num in range(experiment.observation_budget):
        suggestion = conn.experiments(experiment.id).suggestions().create()

        print("Running trial number {}".format(num))
        objective_metric = run_environment(
            h_size=suggestion.assignments['h_size'],
            middle_size=suggestion.assignments['middle_size'],
            lstm_layers=suggestion.assignments['lstm_layers'],
            learning_starts=suggestion.assignments['learning_starts'],
            learning_freq=suggestion.assignments['learning_freq'],
            target_update_freq=suggestion.assignments['target_update_freq'],
            lr=math.exp(suggestion.assignments['log_lr']),
            gamma=suggestion.assignments['gamma'],
            batch_size=suggestion.assignments['batch_size'],
            replay_buffer_size=100000,
            epsilon_decay_steps=suggestion.assignments['epsilon_decay_steps'],
            final_epsilon=0.1,
            root_dir=root_dir,
            num=num
        )

        conn.experiments(experiment.id).observations().create(
            suggestion=suggestion.id,
            value=objective_metric
        )

def test(root_dir):
    for num in range(10):
        print("Running trial number {}".format(num + 1))
        objective_metric = run_environment(
            h_size=38,
            middle_size=128,
            lstm_layers=1,
            learning_starts=100005,
            learning_freq=4,
            target_update_freq=100,
            lr=math.exp(0.01),
            gamma=0.99,
            batch_size=500,
            replay_buffer_size=100000,
            epsilon_decay_steps=50000,
            final_epsilon=0.1,
            root_dir=root_dir,
            num=num
        )
        print("Objective metric: {}".format(objective_metric))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a parameter search with Sigopt')
    parser.add_argument('name', help="name of the search series")
    parser.add_argument('-f', '--force', help="force overwriting of the log folder", action='store_true')
    parser.add_argument('-t', '--training', help="specify custom training file", default="training_sigopt.json")
    args = parser.parse_args()

    os.makedirs("Parameter_search", exist_ok=True)

    root_dir = os.path.join("Parameter_search", args.name)
    if args.force:
        shutil.rmtree(root_dir, ignore_errors=True)
    os.makedirs(root_dir)

    with open(os.path.join(root_dir, "Training_description.json"), 'w') as f:
        json.dump(training, f, indent=4)
    if args.name == "test":
        test(root_dir)
    else:
        search(args.name, root_dir)
