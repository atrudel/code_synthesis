class Agent:
    def __init__(verbose, log_dir):
        self.verbose = verbose
        self.log_dir = log_dir
        if log_dir:
            self.writer = SummaryWriter()
    
    def train(self, reward_func, episodes):
        raise NotImplementedError("You need to implement a train method in your Agent class!")
    
    def act(self, state):
        pass
    
    def assess(self):
        pass
    
    def log_init(self, log_dir):
        if self.verbose:
            print("Starting training [reward = {}, algo = {}] for {} episodes...".format(reward_func.__name__, episodes))
    
    def log(self, log_dir):
        pass
    
    def load(self, path):
        pass
    
    def save(self, path):
        pass
    
    def __del__(self):
        self.writer.close()