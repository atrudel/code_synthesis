import torch

# Hardware usage
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MULTIPROCESSING = True

# Learning parameters
MAX_EPISODES = 50000  # This is just a default value

# Logging parameters
ASSESS_FREQ = 20
LOG_FREQ = ASSESS_FREQ
SAVE_FREQ = 0

# Toy Corewar settings
NUM_ACTIONS = 225
NUM_REGISTERS = 4
MAX_LENGTH = 5
N_INSTRUCTIONS = 4
N_VARS = 3
N_VALS = 20
N_TARGETS = 4
