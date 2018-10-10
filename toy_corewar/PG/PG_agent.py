from agent import Agent
from config import *
from model import VanillaPG
from game.environment import Env
import time
import random
import numpy as np
import torch
import torch.optim as optim

class PG_agent(Agent):
	def __init__(self,
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
                 verbose=False, log_dir=None):
        
		Agent.__init__(self, verbose, log_dir)
		
        # Initialize neural networks
        self.model = VanillaPG(h_size, middle_size, lstm_layers).to(DEVICE)
        
        # Save Policy Gradient parameters
        self.gamma = gamma
        self.batch_size = batch_size
        
        # Initialize Policy Gradient variables
        self.trained = False
        self.loss_function = torch.nn.MSELoss()
        self.optimizer = optim.RMSprop(self.Q.parameters(), lr=lr)
        
        self.verbose = verbose
        self.log_dir = log_dir

        # TODO: make a test for each function 
        def remember(self, state, action, prob, reward):
	         y = np.zeros([self.action_size])
	         y[action] = 1
	         self.gradients.append(np.array(y).astype('float32') - prob)
	         self.states.append(state)
	         self.rewards.append(reward)

	    def act(self, state):
	        state = torch.tensor(state.reshape([1, state.shape[0]]))
	        aprob = self.model.predict(state, self.batch_size).flatten()
	        self.probs.append(aprob.detach())
	        action = np.random.choice(self.action_size, 1 p=prob)[0]
	        return action, prob

	    def discount_rewards(self, rewards):
	        discounted_rewards = np.zeros_like(rewards)
	        running_add = 0
	        for t in reversed(range(0, rewards.size)):
	            if rewards[t] != 0:
	                running_add = 0
	            running_add = runnning_add * self.gamma + rewards[t]
	            discounted_rewards[t] = running_add
	        return discounted_rewards

	    def model_update(self):
	        gradients = np.vstack(self.gradients)
	        rewards = np.vstack(self.rewards)
	        rewards = self.discount_rewards(rewards)
	        rewards = (rewards - np.mean(rewards)) / np.std(rewards - np.mean(rewards))
	        gradients *=rewards
	        X = np.squeeze(np.vstack([self.states]))
	        output = self.model(X)
	        
	        # TODO: make research on model Y value
	        Y = torch.tensor(np.squeeze(np.vstack([gradients]))) 
	        loss = loss_function(output, Y)
	        self.optimizer.zero_grad()
	        self.loss.backward()
	        self.optimizer.step()

	    def train(self):

	        env = Env(reward_func)
		    num_actions = env.action_space_n

		    score = 0
		    episode = 0		  
		    state = env.reset()

		    while True:
		       
		    	# TODO: Check how often to update the weights 
		        action, prob = self.act(state)
		        state, reward, done, info = env.step(action)
		        score += reward
		        self.remember(x, action, prob, reward)

		        if done:
		            episode += 1
		            self.model_update()
		            print('Episode: %d - Score: %f.' % (episode, score))
		            score = 0
		            state = env.reset()

		            if episode > 1 and episode % 10 == 0:
		                self.save('core_war.h5')
