import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../../game')))
from coach import Coach
from NeuralNetworkWrapper import NNetWrapper
from train import Train
from multiprocessing import Pool
from tensorboardX import SummaryWriter
from GOLAI.arena import Arena
import multiprocessing
from game import Game
import numpy as np
import random
import torch
import time
from utils import *

def createRandomOpp():
    
    opponents = []
    value = list(range(0, args.vocabLen))
    for i in range(args.numEps):
        opponents.append(np.random.choice(value, args.predictionLen).tolist())
    
    return opponents

        
def createCoachData(nnet, opponents, iteration):
    random.shuffle(opponents)
    opponents = np.array(opponents)
    opponents = np.split(opponents, args.cpus)
    gpuId = np.arange(args.gpus)
    file_path = os.path.join(args.output_dir, args.time + "/" + str(iteration))
    if iteration % args.saveProgramsInterval == 0:
        os.makedirs(file_path)
    coachData = []
    for i in range(args.cpus):
        coachData.append([nnet, gpuId[0], iteration, opponents[i], file_path])
        gpuId = np.roll(gpuId, 1)
    
    return coachData
    
    
def generateData(nnet, gpuId, iteration, trainingData, file_path):    
    g = Game(Arena(args.gameWidth, args.gameHeight))
    c = Coach(g, nnet, iteration, trainingData, file_path)
    
    return c.learn()



if __name__=="__main__":
    multiprocessing.set_start_method('spawn')
    nnet = NNetWrapper()
    t = Train(nnet)
    writer = SummaryWriter()
    opponents = createRandomOpp()
    
    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    
    if args.load_model:
        print("Load trainExamples from file")
        t.loadTrainExamples()
        
    for i in range(args.numIters):
        coachData = createCoachData(nnet, opponents, i)
        p = Pool(processes = args.cpus)
        data = p.starmap(generateData, coachData)
        p.close()
        
        data = np.array(data)
        nn_input = np.concatenate(data[:, 0])
        opponents = np.concatenate(data[:, 1])
        
        wins = sum(data[:, 2].tolist())
        print(wins)
        writer.add_scalars('zero_10/winning', {'winning': wins}, i)
        nnet = t.run(nn_input.tolist(), i)
        opponents = opponents.tolist()


