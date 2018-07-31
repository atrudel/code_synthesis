from coach import Coach
from model import GolaiZero
from game import Game
from utils import dotdict
import torch

args = dotdict({
    
    #Program
    'numIters': 1000,
    'numEps': 100,
    'vocabWidth': 2, 
    'vocabHeight': 2,
    'programSize': 6,
    'programWidth': 6,
    'programHeight': 6,
    'predictionLen': 6*6 // 2*2,
    'vocabLen': 2**(2*2), 
    
    # Simulations
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 25,
    'arenaCompare': 40,
    'cpuct': 1,
    
    # Game 
    'gameSteps': 100,
    
    # Model
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512,
    'resnetBlocks': 10,
    'resnetInputDepth': 1,
    'resnetChannelDepth': 64,
    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50', 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})

if __name__=="__main__":
    g = Game(args)
    nnet = NNetWrapper(args)
    
    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
        
    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
    
    
    
# Structure from: https://github.com/suragnair/alpha-zero-general/blob/master/main.py