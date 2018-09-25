from DQN_utils import *
from config import *
from model import Dueling_DQN
import random
import inspect
import os
import time, datetime
from toyCorewar import ToyCorewar
from environment import Env
from program_synthesis import Program, Instruction
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

def train_DQN(
    reward_func, 
    episodes,
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
    verbose=False, 
    log_dir=None
):
    
    env = Env(reward_func)
    num_actions = env.action_space_n
    best_score = -float('Inf')
    best_episode = 0
    best_Q = Dueling_DQN(h_size, middle_size, lstm_layers)
   
    epsilon_schedule = LinearSchedule(schedule_episodes=epsilon_decay_steps, final_p=0.1)
    replay_buffer = deque(maxlen=replay_buffer_size)
    Q = Dueling_DQN(h_size, middle_size, lstm_layers)
    Q_target = Dueling_DQN(h_size, middle_size, lstm_layers)
    Q_target.load_state_dict(Q.state_dict())
    Q.to(DEVICE)
    Q_target.to(DEVICE)
    
    loss_function = torch.nn.MSELoss()
    optimizer = optim.RMSprop(Q.parameters(), lr=lr)
    num_parameter_updates = 0
    writer = SummaryWriter()

    if verbose:
        print("Starting training [reward function = {}] for {} episodes...".format(
            reward_func.__name__, episodes))
    
    if log_dir is not None:
        log_file = os.path.join(log_dir, "logs")
        with open(log_file, "w") as f:
            print("Starting training for {} episodes...".format(episodes), file=f)
            print("Reward function:\n\n{}\n\n\n".format(inspect.getsource(reward_func)), file=f)
        model_dir = os.path.join(log_dir, "models")
        os.makedirs(model_dir)
    
    start_time = time.time()
    for episode in range(episodes):
        s = env.reset()

        for t in range(MAX_LENGTH):
            # Select action with E-greedy policy
            if episode < learning_starts or np.random.rand() < epsilon_schedule.value(episode):
                a = np.random.randint(num_actions)
            else:
                a = Q(state_to_tensors(s)).argmax(1).item()

            # Submit chosen action to the environment
            s_prime, reward, done, info = env.step(a)

            # Store the effect of the action
            replay_buffer.append(Transition(s, a, reward, s_prime, done))

            # New state becomes current state
            s = s_prime

            # EXPERIENCE REPLAY (every LEARNING_FREQth time step after learning starts) 
            if (episode > learning_starts and (episode * MAX_LENGTH + t) % learning_freq == 0):
                # Sample from the replay buffer
                transitions = random.sample(replay_buffer, batch_size)

                # Extract each batch of elements from the sample of transitions
                batch = Transition(*zip(*transitions))
                state_batch = batch_to_tensors(batch.state)
                action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(DEVICE)
                reward_batch = torch.tensor(batch.reward, dtype=torch.float).to(DEVICE)
                next_state_batch = batch_to_tensors(batch.next_state)
                done_batch = torch.tensor(batch.done, dtype=torch.float).to(DEVICE)

                # Get the current network's estimations for the q-values of all (state, action)
                # pairs in the batch
                q_s_a = Q(state_batch).gather(1, action_batch).squeeze()

                # Calculate the corresponding target q-values to send to the loss function
                a_prime =  Q(next_state_batch).argmax(1).unsqueeze(1)
                q_s_a_prime = Q_target(next_state_batch).gather(1, a_prime).squeeze()
                q_s_a_prime *= 1 - done_batch
                target_q_s_a = reward_batch + gamma * q_s_a_prime
                target_q_s_a = target_q_s_a.detach()

                # Backprop
                loss = loss_function(q_s_a, target_q_s_a)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                num_parameter_updates += 1

                # Update target DQN every once in a while
                if num_parameter_updates % target_update_freq == 0:
                    Q_target.load_state_dict(Q.state_dict())

            if done:
                break
        
        # Console output
        if verbose and (episode + 1) % LOG_FREQ == 0:
            print("Episode {} completed for {}".format(episode + 1, reward_func.__name__))
        
        # Log output
        if log_dir is not None and (episode + 1) % LOG_FREQ == 0:
            with open(log_file, "a") as f:
                current_time = datetime.timedelta(seconds=(time.time()-start_time))
                print("Episode {}: [time:  {}]\n".format(episode+1, str(current_time)), file=f)
                score = assess(Q, reward_func, file=f)
                print("\n\n\n", file=f)
                # log to Tensorboard
                writer.add_scalars(log_dir, {'rewards': score}, episode)
                
        # Agent assessment
        if (episode + 1) % ASSESS_FREQ == 0:
            score = assess(Q, reward_func, print=False)
            if score > best_score:
                best_score = score
                best_episode = episode + 1
                best_Q.load_state_dict(Q.state_dict())
        
    writer.close()
    return best_Q, best_score, best_episode