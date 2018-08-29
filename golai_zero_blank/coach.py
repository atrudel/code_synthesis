import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../../game/')))
from GOLAI.arena import Arena
from collections import deque
from MCTS import MCTS
import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
from turn_program_into_file import turn_program_into_file
import time
from utils import *
import os
import sys
from pickle import Pickler, Unpickler
from random import shuffle
import random
from copy import deepcopy


class Coach():

    def __init__(self, game, nnet, iteration, opponents, file_path, gpuId, cpuId):
        self.cpuId = cpuId
        self.gpuId = gpuId
        self.game = game
        self.nnet = nnet
        self.savePrograms = False
        self.episode = 0
        self.iteration = iteration
        self.file_path = file_path
        self.mcts = MCTS(self.game, self.nnet)
        self.trainOpponents = opponents.tolist()
        self.nextOpponents = []
        self.wins = 0


    def dirichlet_noise(self, probs):

        dim = (len(probs))
        probs = np.array(probs, dtype=float)
        new_probs = (1 - EPS) * probs + EPS * \
        np.random.dirichlet(np.full(dim, ALPHA))

        return new_probs
    
    def print_winner(self, value):
        if value == 1.0:
            return("win")
        elif value == -1.0:
            return("lost")
        else:
            return("draw")
    
    def executeEpisode(self):

        trainExamples = []
        self.curProgram = self.game.getInitProgram()
        self.curOpponent = self.trainOpponents[self.episode]
        episodeStep = 0

        while True:
            episodeStep += 1
            temp = int(episodeStep < TEMP_THRESHOLD)
            
            iterations = PREDICTION_LEN - (episodeStep - 1)
            pi = self.mcts.getActionProb(self.curProgram, iterations, self.gpuId, temp)
            
            if DIRICHLET_NOISE and temp:
                pi = self.dirichlet_noise(pi)
                    
            trainExamples.append([self.game.neuralNetworkInput(self.curProgram), pi, None])
            action = np.random.choice(len(pi), p=pi)
            self.game.getNextState(self.curProgram, action)

            if episodeStep == PREDICTION_LEN:
                self.nextOpponents.append(self.curProgram)
                r, ones, twos, p1, p2 = self.game.getGameEnded(self.curProgram, self.curOpponent)
                self.wins += r
                if self.savePrograms:
                    turn_program_into_file(p1, self.file_path + '/' + str(self.cpuId) + '-' + str(self.episode) +
                                           '-' + self.print_winner(r) + "-tiles-" + str(ones) + "-p1.rle")
                    turn_program_into_file(p2, self.file_path + '/' + str(self.cpuId) + '-' + str(self.episode) +
                                           '-' + self.print_winner(-r) + "-tiles-" + str(twos) + "-p2.rle")

                return[(x[0], x[1], r) for x in trainExamples]

    def learn(self):
        
        if self.iteration % SAVE_PROGRAMS_INTERVAL == 0:
            self.savePrograms = True
        else:
            self.savePrograms = False
        self.wins = 0
        
        self.mcts = MCTS(self.game, self.nnet)
        
        iterationTrainExamples = []
        for eps in range(len(self.trainOpponents)):
            self.episode = eps
            if eps % 10 == 0:
                self.mcts = MCTS(self.game, self.nnet)
            iterationTrainExamples += self.executeEpisode()
            
        return iterationTrainExamples, self.nextOpponents, self.wins
               

