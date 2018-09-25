import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../../game')))
from coach import Coach
from NeuralNetworkWrapper import NNetWrapper
from train import Train
from tensorboardX import SummaryWriter
from GOLAI.arena import Arena
import multiprocessing
from multiprocessing import Pool
import multiprocessing
from game import Game
from copy import deepcopy
import numpy as np
import random
import torch
import time
from utils import *
from random import randint


def createNCompetitors():
    episode_opponents = []
    for i in range(NUM_EPS):
        episode_opponents.append(np.random.choice(VOCAB, PREDICTION_LEN).tolist())
    return episode_opponents

def createRandomOpp():
    
    opponents = []
    for i in range(NUM_ITERS_FOR_TRAIN_EXAMPLES_HISTORY):
        episode_opponents = []
        opponents.append(createNCompetitors())
    return opponents

def createOpponents(opponents):
    random_opponents = []
    for i in range(NUM_EPS):
        one_game = []
        for n in range(NUM_COMPETITORS):
            stack = randint(0, NUM_ITERS_FOR_TRAIN_EXAMPLES_HISTORY - 1)
            idx = randint(0, NUM_EPS - 1)
            one_game.append(opponents[stack][idx])
        random_opponents.append(one_game)
    
    random_opponents = np.array(random_opponents)
    random_opponents = np.split(random_opponents, CPUS)
    
    return random_opponents


def createCoachData(nnet, opponents, iteration):
    opponents_data = deepcopy(opponents)
    opponents_data.append(createNCompetitors())
    opponents_data = createOpponents(opponents_data)
    gpuId = np.arange(GPUS)
    file_path = os.path.join(OUTPUT_DIR, TIME + "/" + str(iteration))
    if iteration % SAVE_PROGRAMS_INTERVAL == 0:
        os.makedirs(file_path)
    coachData = []
    for i in range(CPUS):
        cpuId = i
        coachData.append([nnet, gpuId[0], cpuId, iteration, opponents_data[i], file_path])
        gpuId = np.roll(gpuId, 1)
    
    return coachData
    
    
def generateData(nnet, gpuId, cpuId, iteration, trainingData, file_path):
    with torch.cuda.device(int(gpuId)):
        gnet = deepcopy(nnet)
    
    g = Game(Arena(GAME_WIDTH, GAME_HEIGHT))
    c = Coach(g, gnet, iteration, trainingData, file_path, gpuId, cpuId)

    return c.learn()



if __name__=="__main__":
    multiprocessing.set_start_method('spawn')
    nnet = NNetWrapper()
    t = Train(nnet)
    writer = SummaryWriter()
    opponents = createRandomOpp()
    opponents.pop(0)
    
    if LOAD_MODEL:
        nnet.load_checkpoint(LOAD_FOLDER_FILE[0], LOAD_FOLDER_FILE[1])
    
    if LOAD_MODEL:
        print("Load trainExamples from file")
        t.loadTrainExamples()
        
    for i in range(NUM_ITERS):
        # Distribute input data to CPUS and create training data
        coachData = createCoachData(nnet, opponents, i)
        p = Pool(processes = CPUS)
        data = p.starmap(generateData, coachData)
        p.close()
        
        # Organise training data
        data = np.array(data)
        nn_input = np.concatenate(data[:, 0])
        
        opponents.pop(0)
        latest_opponents = np.concatenate(data[:, 1])
        opponents.append(latest_opponents.tolist())
                
        wins = sum(data[:, 2].tolist())
        print(wins)
        writer.add_scalars('zero_10/winning', {'winning': wins}, i)
        
        # Train Network
        nnet = t.run(nn_input.tolist(), i)


