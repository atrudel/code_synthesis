from DQN.model import Dueling_DQN
from DQN.DQN_utils import *
from agent import Agent
from game.environment import Env
import time
import random
import config
import numpy as np
import torch
import torch.optim as optim
from task_manager import Task_Manager

CFG = config.get_cfg()
CWCFG = CFG.settings.toy_corewar

class DQN_Agent(Agent):
    def __init__(self,
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
                 verbose=False, log_dir=None):
        
        Agent.__init__(self, verbose, log_dir)
        
        # Initialize neural networks
        self.Q = Dueling_DQN(h_size, middle_size, lstm_layers).to(DEVICE)
        self.Q_target = Dueling_DQN(h_size, middle_size, lstm_layers).to(DEVICE)
        self.model = self.Q # alias which is used in the parent class
        self.best_model = Dueling_DQN(h_size, middle_size, lstm_layers)
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.best_model.load_state_dict(self.Q.state_dict())
        
        # Save Q-learning parameters
        self.learning_starts = learning_starts if (learning_starts > batch_size / CWCFG.MAX_LENGTH) else batch_size
        self.learning_freq = learning_freq
        self.target_update_freq = target_update_freq
        self.gamma = gamma
        self.batch_size = batch_size
        
        # Initialize Q-learning variables
        self.loss_function = torch.nn.MSELoss()
        self.optimizer = optim.RMSprop(self.Q.parameters(), lr=lr)
        self.num_parameter_updates = 0
        self.epsilon_decay_steps = epsilon_decay_steps
        self.final_epsilon = final_epsilon
        self.replay_buffer = deque(maxlen=replay_buffer_size)



    def act(self, state, episode=None, e_greedy=False):
        '''Returns an action based on a state. 
        An epsilon-greedy policy requires the episode to be specified'''

        if e_greedy and (episode < self.learning_starts or 
                         np.random.rand() < self.epsilon_schedule.value(episode)):
            return np.random.randint(CFG.settings.toy_corewar.NUM_ACTIONS)
        else:
            return self.Q(state_to_tensors(state)).argmax(1).item()


    def train(self, Reward_func, reward_settings, episodes, targets=None, reg_inits=None):
        if self.verbose:
            print("Starting training [algo = {}, reward = {}] for {} episodes...".format(
                    self.__class__.__name__, Reward_func.__class__.__name__, episodes))

        self.tasks = Task_Manager(Reward_func, reward_settings, targets, reg_inits, episodes)
        self.epsilon_schedule = LinearSchedule(self.epsilon_decay_steps, episodes, self.final_epsilon)
        self.start_time = time.time()
        for episode in range(episodes):
            reward_func, reg_init = self.tasks.get_current(episode)
            env = Env(reward_func)
            s = env.reset(reg_init)
            
            for t in range(CWCFG.MAX_LENGTH):
                # Select action with E-greedy policy
                a = self.act(s, episode=self.total_episodes, e_greedy=True)
                
                # Submit chosen action to the environment
                s_prime, reward, done, _ = env.step(a)
                
                # Store the effect of the action
                self.replay_buffer.append(Transition(s, a, reward, s_prime, done))
                
                # New state becomes current state
                s = s_prime
                
                # EXPERIENCE REPLAY (every LEARNING_FREQth time step after learning starts) 
                if (self.total_episodes > self.learning_starts and
                    len(self.replay_buffer) >= self.batch_size and
                    (episode * CWCFG.MAX_LENGTH + t) % self.learning_freq == 0):
                    self.experience_replay()
                
                if done:
                    break

            # Assess agent performance (and keep track of the best one)
            if (episode) % CFG.settings.ASSESS_FREQ == 0:
                if targets is not None:
                    self.evaluate(log=(episode % CFG.settings.LOG_FREQ == 0))
                else:
                    self.generalize(Reward_func, 100, reward_settings, log=(episode % CFG.settings.LOG_FREQ == 0))
            # Save best model periodically
            if CFG.settings.SAVE_FREQ > 0 and (episode) % CFG.settings.SAVE_FREQ == 0:
                self.save("best", best=True)
                # self.save("Episode_{}".format(episode))

            self.total_episodes += 1

                    
    def experience_replay(self):
        # Sample from the replay buffer
        transitions = random.sample(self.replay_buffer, self.batch_size)
        
        # Extract each batch of elements from the sample of transitions
        batch = Transition(*zip(*transitions))
        state_batch = batch_to_tensors(batch.state)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(DEVICE)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float).to(DEVICE)
        next_state_batch = batch_to_tensors(batch.next_state)
        done_batch = torch.tensor(batch.done, dtype=torch.float).to(DEVICE)   
        
        # Get the current network's estimations for the q-values of all (state, action)
        # pairs in the batch
        q_s_a = self.Q(state_batch).gather(1, action_batch).squeeze()
        
        # Calculate the corresponding target q-values to send to the loss function
        a_prime =  self.Q(next_state_batch).argmax(1).unsqueeze(1)
        q_s_a_prime = self.Q_target(next_state_batch).gather(1, a_prime).squeeze()
        q_s_a_prime *= 1 - done_batch
        target_q_s_a = reward_batch + self.gamma * q_s_a_prime
        target_q_s_a = target_q_s_a.detach()
        
        # Backprop
        loss = self.loss_function(q_s_a, target_q_s_a)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.num_parameter_updates += 1
        if self.writer is not None:
            self.writer.add_scalars(self.log_dir, {self.loss_function.__class__.__name__: loss}, self.total_episodes)

        # Update target DQN every once in a while
        if self.num_parameter_updates % self.target_update_freq == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())
                
