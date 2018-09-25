import sigopt
from reward import *
import torch
import math
from environment import Env
from train import train_DQN


MAX_EPISODES = 100000


def search(reward_func):
    conn = sigopt.Connection(client_token='QWILQOQCAHZIVDIMLOLIQZGYCBTIKTVTEFQMJXSAVWQVBLTA')
    
    DQN_experiment = conn.experiments().create(
        name= reward_func.__name__ + 'Dueling_DQN',
        observation_budget=300,
        parameters=[
            dict(name='h_size', type='int', bounds=dict(min=20,max=100)),
            dict(name='middle_size', type='int', bounds=dict(min=30, max=250)),
            dict(name='lstm_layers', type='int', bounds=dict(min=1, max=2)),
            dict(name='epsilon_decay_steps', type='int', bounds=dict(min=MAX_EPISODES/10, max=MAX_EPISODES)),
            dict(name='learning_starts', type='int', bounds=dict(min=0, max=MAX_EPISODES/2)),
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
        
        print("Running trial number {}".format(num+1))
        objective_metric = run_environment(
            h_size=suggestion.assignments['h_size'],
            middle_size=suggestion.assignments['middle_size'],
            lstm_layers=suggestion.assignments['lstm_layers'],
            epsilon_decay_steps=suggestion.assignments['epsilon_decay_steps'],
            learning_starts=suggestion.assignments['learning_starts'],
            learning_freq=suggestion.assignments['learning_freq'],
            target_update_freq=suggestion.assignments['target_update_freq'],
            lr=math.exp(suggestion.assignments['log_lr']),
            gamma=suggestion.assignments['gamma'],
            batch_size=suggestion.assignments['batch_size'],
            replay_buffer_size=100000,
            reward_func=reward_func
        )
        
        conn.experiments(experiment.id).observations().create(
            suggestion=suggestion.id,
            value=objective_metric
        )

def run_environment(
    h_size,
    middle_size,
    lstm_layers,
    epsilon_decay_steps,
    learning_starts,
    learning_freq,
    target_update_freq,
    lr,
    gamma,
    batch_size,
    replay_buffer_size,
    reward_func
):
    DQN, reward, best_episode = train_DQN(reward_func, 
                                          MAX_EPISODES, 
                                          h_size, 
                                          middle_size, 
                                          lstm_layers, 
                                          epsilon_decay_steps, 
                                          learning_starts, 
                                          learning_freq, 
                                          target_update_freq, 
                                          lr, 
                                          gamma, 
                                          batch_size, 
                                          replay_buffer_size)
    return reward + 1 / best_episode
    
    
            
        
if __name__=="__main__":
    search(specific_register_values)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    