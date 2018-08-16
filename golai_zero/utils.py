import time
import torch
class dotdict(dict):
    def __getattr__(self, name):
        return self[name]


PROGRAM_WIDTH = 6
PROGRAM_HEIGHT = 6
PROGRAM_SIZE = 6
VOCAB_LEN = 2**(2*2)
CUDA = torch.cuda.is_available()
EPOCHS = 20
BATCH_SIZE = 1000
RESNET_BLOCKS = 10
RESNET_INPUT_DEPTH = 2
RESNET_CHANNEL_DEPTH = 64

args = dotdict({
    
    #Hardware
    'cpus': 5,
    'gpus': 2,
    
    #Program
    'numIters': 100000,
    'numEps': 20,
    'vocabWidth': 2, 
    'vocabHeight': 2,
    'programSize': 6,
    'programWidth': 6,
    'programHeight': 6,
    'predictionLen': (6*6) // (2*2),
    'vocabLen': 2**(2*2), 
    
    # Simulations
    'tempThreshold': 5,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 16,
    'arenaCompare': 40,
    'cpuct': 1,
    'eps': 0.25,
    'alpha': 0.3, # 0.03 for Go 0.3 for Chess
    'dirichlet_noise': True,
    
    # Game 
    'gameSteps': 100,
    'gameHeight': 27, 
    'gameWidth': 27,
    
    # Model
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 20,
    'batch_size': 1000,
    'cuda': torch.cuda.is_available(),
    'resnetBlocks': 10,
    'resnetInputDepth': 2,
    'resnetChannelDepth': 64,
    'checkpoint': './checkpoint/',
    'load_model': False,
    
    # Save and Load
    'savePrograms': True,
    'saveProgramsInterval': 5,
    'time': str(int(time.time())),
    'output_dir': "./output/",
    'program_dir': "",
    'load_folder_file': ('./checkpoint/', 'checkpoint_4.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})