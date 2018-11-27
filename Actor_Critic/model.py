import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import config
import numpy as np

CFG = config.get_cfg()
CWCFG = CFG.settings.toy_corewar

torch.set_default_tensor_type('torch.FloatTensor')

class AC_Model(nn.Module):
    def __init__(self,  h_size, middle_size, lstm_layers):
        super(AC_Model, self).__init__()
        self.num_lstm_layers = lstm_layers

        input_size = CWCFG.N_INSTRUCTIONS + CWCFG.N_VARS * CWCFG.NUM_REGISTERS + 1
        s_size = CWCFG.N_TARGETS * 2

        self.lstm_p = nn.LSTM(input_size=input_size, hidden_size=h_size, num_layers=lstm_layers)
        self.fc_s1 = nn.Linear(in_features=s_size, out_features=s_size)
        self.fc_s2 = nn.Linear(in_features=s_size, out_features=s_size)

        self.fc1 = nn.Linear(in_features=(h_size + s_size), out_features=middle_size)

        self.fc_actor = nn.Linear(in_features=middle_size, out_features=CWCFG.NUM_ACTIONS)
        self.fc_critic = nn.Linear(in_features=middle_size, out_features=1)

        self.relu = nn.ReLU()

    def forward(self, state):
        p, s = state

        # Collecting the hidden state
        _,(p,_) = self.lstm_p(p.float())

        # Process state vector in 2 FC layers
        s = self.relu(self.fc_s1(s.float()))
        s = self.relu(self.fc_s2(s.float()))

        # Concatenate P and S vectors and process in 2 FC layers
        x = torch.cat((p[self.num_lstm_layers - 1], s), dim=1)
        x = self.relu(self.fc1(x))

        # Split processing in 2 streams: actor and critic
        action_scores = F.softmax(self.fc_actor(x), dim=-1)
        state_value = self.fc_critic(x)

        return action_scores, state_value
