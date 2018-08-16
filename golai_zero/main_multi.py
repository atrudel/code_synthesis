
# coding: utf-8

# In[1]:


import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../../game')))
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from coach import Coach
from NeuralNetworkWrapper import NNetWrapper
from multiprocessing import Pool
from tensorboardX import SummaryWriter
from GOLAI.arena import Arena
from game import Game
from utils import dotdict
import torch
import time

args = dotdict({
    
    #Hardware
    'cpus': 2,
    'gpus': 2,
    
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

def createRandomOpp(args):
    
    opponents = []
    value = list(range(0, args.vocabLen))
    for i in range(args.numEps):
        opponents.append(np.random.choice(value, args.predictionLen).tolist())
    
    return opponents

        
def createCoachData(nnet, args, opponents, iteration):
    opponents.shuffle()
    opponents = np.array(opponents)
    np.split(opponents, args.cpus)
    
    gpuId = np.arange(args.gpus)
    coachData = []
    for i in range(args.cpus):
        coachData.append([nnet, gpuId, iteration, args, opponents[i]])
        np.roll(gpus, 1)
    
    return coachData
    
    
def generateData(variables):    
    nnet = variables[0]
    gpuId = variables[1]
    iteration = variables[2]
    args = variables[3]
    trainingData = variables[2]
    
    g = Game(args, Arena(args.gameWidth, args.gameHeight))
    c = Coach(g, nnet, args, opponents)
    
    return c.learn()



if __name__=="__main__":
    args.program_dir = os.path.join(args.output_dir, args.time)
    g = Game(args, Arena(args.gameWidth, args.gameHeight))
    nnet = NNetWrapper(args)
    t = Train(g, nnet, args)
    writer = SummaryWriter()
    opponents = createRandomOpp(args)
    
    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    
    if args.load_model:
        print("Load trainExamples from file")
        t.loadTrainExamples()
        
    for i in range args.numIters:
        coachData = createCoachData(nnet, args, opponents, i)
        os.makedirs(os.path.join(args.program_dir, str(i)))
        p = Pool(processes = args.cpus)
        data = p.starmap(generateData, coachData)
        p.close()
        
        data = np.array(data)
        wins = sum(data[:, 2].tolist())
       
        writer.add_scalars('zero_10/winning', {'winning': wins}, i)
        nnet = t.run(data[0])
        opponents = data[:, 1]
