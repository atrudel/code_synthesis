from agent import Agent
from Actor_Critic.model import AC_Model
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from game.environment import Env
import config
from config import DEVICE
from DQN.DQN_utils import state_to_tensors
from task_manager import Task_Manager

CFG = config.get_cfg()
CWCFG = CFG.settings.toy_corewar

class AC_Agent(Agent):
    def __init__(self,
                 h_size,
                 middle_size,
                 lstm_layers,
                 lr,
                 gamma,
                 verbose=False, log_dir=None):
        Agent.__init__(self, verbose, log_dir)
        self.model = AC_Model(h_size, middle_size, lstm_layers)
        self.best_model = AC_Model(h_size, middle_size, lstm_layers)
        self.best_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self.gamma = gamma

        self.saved_actions = []
        self.rewards = []

    def act(self, state):
        probs, state_value = self.model(state_to_tensors(state))
        m = Categorical(probs)
        action = m.sample()
        self.saved_actions.append((m.log_prob(action), state_value))
        return action.item()

    def train(self, Reward_func, reward_settings, episodes, targets, reg_inits):
        if self.verbose:
            print("Starting training [algo = {}, reward = {}] for {} episodes...".format(
                    self.__class__.__name__, Reward_func.__name__, episodes))

        def update_model():
            eps = np.finfo(np.float32).eps.item()
            R = 0
            saved_actions = self.saved_actions
            policy_losses = []
            value_losses = []
            rewards = []

            for r in self.rewards[::-1]:
                R = r + self.gamma * R
                rewards.insert(0, R)
            rewards = torch.tensor(rewards)
            rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
            for (log_prob, value), r in zip(saved_actions, rewards):
                advantage = r - value.item()
                policy_losses.append(-log_prob * advantage)
                value_losses.append(F.smooth_l1_loss(value.squeeze(1), torch.tensor([r])))
            self.optimizer.zero_grad()
            loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
            loss.backward()
            self.optimizer.step()
            del self.rewards[:]
            del self.saved_actions[:]

        self.tasks = Task_Manager(Reward_func, reward_settings, targets, reg_inits, episodes)
        self.start_time = time.time()

        for i_episode in range(episodes):
            reward_func, reg_init = self.tasks.get_current(i_episode)
            env = Env(reward_func)
            state = env.reset(reg_init)

            for it in range(CWCFG.MAX_LENGTH):
                action = self.act(state)
                state, reward, done, _ = env.step(action)

                self.rewards.append(reward)
                if done:
                    break

            update_model()

            # Assess agent performance (and keep track of the best one)
            if (i_episode) % CFG.settings.ASSESS_FREQ == 0:
                self.evaluate(log=(i_episode % CFG.settings.LOG_FREQ == 0))

            # Save best model periodically
            if CFG.settings.SAVE_FREQ > 0 and (i_episode) % CFG.settings.SAVE_FREQ == 0:
                self.save("best", best=True)
                # self.save("Episode_{}".format(episode))
            self.total_episodes += 1