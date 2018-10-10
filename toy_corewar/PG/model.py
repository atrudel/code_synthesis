import torch
import torch.nn as nn
import torch.optim as optim
from config import *

torch.set_default_tensor_type('torch.FloatTensor')

class VanillaPG(nn.Module):
    def __init__(self):
        super(Dueling_DQN, self).__init__()
        
        h_size_a = 10
        h_size_b = 20
        h_size_c = 5
        s_size = N_TARGETS * 2
        
        self.lstm_p_a = nn.LSTM(input_size=N_INSTRUCTIONS, hidden_size=h_size_a, num_layers=2)
        self.lstm_p_b = nn.LSTM(input_size=(N_VARS * NUM_REGISTERS), hidden_size=h_size_b, num_layers=2)
        self.lstm_p_c = nn.LSTM(input_size=1, hidden_size=h_size_c, num_layers=2)
        self.fc_s1 = nn.Linear(in_features=s_size, out_features=s_size)
        self.fc_s2 = nn.Linear(in_features=s_size, out_features=s_size)
        
        self.fc1 = nn.Linear(in_features=(h_size_a + h_size_b + h_size_c + s_size), out_features=200)
        self.fc2 = nn.Linear(in_features=200, out_features=128)
        
        self.fc1_adv = nn.Linear(in_features=128, out_features=128)
        self.fc1_val = nn.Linear(in_features=128, out_features=128)
        self.fc2_adv = nn.Linear(in_features=128, out_features=NUM_ACTIONS)
        self.fc2_val = nn.Linear(in_features=128, out_features=1)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
    
    def forward(self, state):
        p_a, p_b, p_c, s = state
        
        _,(p_a,_) = self.lstm_p_a(p_a.float())
        _,(p_b,_) = self.lstm_p_b(p_b.float())
        _,(p_c,_) = self.lstm_p_c(p_c.float())
        
        s = self.relu(self.fc_s1(s.float()))
        s = self.relu(self.fc_s2(s.float()))

        x = torch.cat((p_a[1], p_b[1], p_c[1], s), dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        
        x = self.softmax(x)
        return x