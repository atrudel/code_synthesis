from DQN.model import Dueling_DQN
from agent import Agent

class DQN_Agent(Agent):
    def __init__(h_size, 
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
        self.Q = Dueling_DQN(h_size, middle_size, lstm_layers).to(DEVICE)
        self.Q_target = Dueling_DQN(h_size, middle_size, lstm_layers).to(DEVICE)
        self.best_model = Dueling_DQN(h_size, middle_size, lstm_layers)
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.best_model.load_state_dict(self.Q.state_dict())
        
        # Save Q-learning parameters
        self.epsilon_decay_steps = epsilon_decay_steps
        self.learning_starts = learning_starts
        self.learning_freq = learning_freq
        self.target_update_freq = target_update_freq
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        
        # Initialize Q-learning variables
        self.replay_buffer = deque(maxlen=replay_buffer_size)    
        self.best_score = -float('Inf')
        self.best_episode = 0
        
        self.verbose = verbose
        self.log_dir = log_dir
        

    def train(self, reward_func, episodes):
        env = Env(reward_func)
        epsilon_schedule = LinearSchedule(schedule_episodes=epsilon_decay_steps, final_p=0.1)
       
        loss_function = torch.nn.MSELoss()
        optimizer = optim.RMSprop(Q.parameters(), lr=lr)
        num_parameter_updates = 0
        
        