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
import os
import sys
from pickle import Pickler, Unpickler
from random import shuffle
import random
from tensorboardX import SummaryWriter
from copy import deepcopy
from utils import *

class Train():

    def __init__(self, nnet):
        self.nnet = nnet
        self.episode = 0
        self.file_path = ""
        self.wins = 0
        self.trainExamplesHistory = []

    def run(self, iterationTrainExamples, i):

        self.trainExamplesHistory.append(iterationTrainExamples)

        if len(self.trainExamplesHistory) > NUM_ITERS_FOR_TRAIN_EXAMPLES_HISTORY:
            print("len(trainExamplesHistory) =", len(self.trainExamplesHistory),
              " => remove the oldest trainExamples")
            self.trainExamplesHistory.pop(0)

        self.saveTrainExamples(i-1)
        trainExamples = []

        for e in self.trainExamplesHistory:
            trainExamples.extend(e)
        shuffle(trainExamples)

        self.nnet.save_checkpoint(folder=CHECKPOINT, filename=self.getCheckpointFile(i))
        self.nnet.train(trainExamples)

        return self.nnet

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = CHECKPOINT

        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration)+".examples")

        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(LOAD_FOLDER_FILE[0], LOAD_FOLDER_FILE[1])
        examplesFile = modelFile+".examples"

        if not os.path.isfile(examplesFile):
            print(examplesFile)
            r = input("File with trainExamples not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with trainExamples found. Read it.")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            f.closed
            self.skipFirstSelfPlay = True