
# coding: utf-8

# In[1]:


import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../../game')))
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from coach import Coach
from NeuralNetworkWrapper import NNetWrapper
from GOLAI.arena import Arena
from game import Game
from utils import dotdict
import torch
import time

args = dotdict({
    
    #Program
    'numIters': 100000,
    'numEps': 1000,
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
    'num_channels': 512,
    'resnetBlocks': 10,
    'resnetInputDepth': 1,
    'resnetChannelDepth': 64,
    'checkpoint': './checkpoint/',
    'load_model': False,
    
    # Save and Load
    'savePrograms': True,
    'time': str(int(time.time())),
    'output_dir': "./output/",
    'load_folder_file': ('./checkpoint/', 'checkpoint_4.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})

if __name__=="__main__":
    g = Game(args, Arena(args.gameWidth, args.gameHeight))
    nnet = NNetWrapper(args)
    
    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
        
    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
    
# Structure from: https://github.com/suragnair/alpha-zero-general/blob/master/main.py

