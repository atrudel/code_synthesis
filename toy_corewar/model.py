import torch
import torch.nn as nn
import torch.optim as optim
from config import *

torch.set_default_tensor_type('torch.FloatTensor')

class Dueling_DQN(nn.Module):
    def __init__(self):
        super(Dueling_DQN, self).__init__()
        
        h_size_a = 50
        h_size_b = 100
        h_size_c = 10
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
    
    def forward(self, state):
        p_a, p_b, p_c, s = state
        
        # Process instruction, variable and value embeddings
        # in separate streams of 2-layer LSTMs
        # Collecting the hidden state
        _,(p_a,_) = self.lstm_p_a(p_a.float())
        _,(p_b,_) = self.lstm_p_b(p_b.float())
        _,(p_c,_) = self.lstm_p_c(p_c.float())
        
        # Process state vector in 2 FC layers
        s = self.relu(self.fc_s1(s.float()))
        s = self.relu(self.fc_s2(s.float()))
        
        # Concatenate P and S vectors and process in 2 FC layers
        x = torch.cat((p_a[1], p_b[1], p_c[1], s), dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        
        # Split processing in 2 streams: value and advantage
        adv = self.relu(self.fc1_adv(x))
        val = self.relu(self.fc1_val(x))
        
        adv = self.fc2_adv(adv)
        val = self.fc2_val(val).expand(-1, NUM_ACTIONS)
        
        x = val + adv - adv.mean().expand(NUM_ACTIONS)
        
        return x