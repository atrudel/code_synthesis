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


class Coach():

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.episode = 0
        self.file_path = ""
        self.writer = SummaryWriter()
        #self.pnet = self.nnet.__class__(self.game)
        self.args = args
        self.program_dir = os.path.join(self.args.output_dir, self.args.time)


        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []
        self.nextOpponents = []
        self.wins = 0

    def createRandomOpp(self):

        for i in range(self.args.numEps):
            value = list(range(0, self.args.vocabLen))
            self.nextOpponents.append(np.random.choice(value, self.args.predictionLen).tolist())

    def dirichlet_noise(self, probs):

        dim = (len(probs))
        probs = np.array(probs, dtype=float)
        new_probs = (1 - self.args.eps) * probs + self.args.eps * \
        np.random.dirichlet(np.full(dim, self.args.alpha))

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
            temp = int(episodeStep < self.args.tempThreshold)
            
            iterations = self.args.predictionLen - (episodeStep - 1)
            pi = self.mcts.getActionProb(self.curProgram, self.curOpponent, iterations, temp)
            
            if self.args.dirichlet_noise and temp:
                pi = self.dirichlet_noise(pi)
                    
            trainExamples.append([self.game.neuralNetworkInput(self.curProgram, self.curOpponent), pi, None])
            action = np.random.choice(len(pi), p=pi)
            self.game.getNextState(self.curProgram, action)

            if episodeStep == self.args.predictionLen:
                self.nextOpponents.append(self.curProgram)
                r, ones, twos, p1, p2 = self.game.getGameEnded(self.curProgram, self.curOpponent)
                self.wins += r
                if self.args.savePrograms:
                    turn_program_into_file(p1, self.file_path + str(self.episode) +
                                           '-' + self.print_winner(r) + "-tiles-" + str(ones) + "-p1.rle")
                    turn_program_into_file(p2, self.file_path + str(self.episode) +
                                           '-' + self.print_winner(-r) + "-tiles-" + str(twos) + "-p2.rle")

                return[(x[0], x[1], r) for x in trainExamples]

    def learn(self):

        self.createRandomOpp()
        os.makedirs(self.program_dir)

        for i in range(1, self.args.numIters+1):

            print('---------Iter ' + str(i) + '---------')


            iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

            self.file_path = os.path.join(self.program_dir, str(i))
            self.file_path += "/"
            eps_time = AverageMeter()
            bar = Bar('Self Play', max=self.args.numEps)
            end = time.time()
            self.trainOpponents = deepcopy(self.nextOpponents)
            self.nextOpponents = []
            shuffle(self.trainOpponents)
            if i % 5 == 0:
                self.args.savePrograms = True
                os.makedirs(self.file_path)
            else:
                self.args.savePrograms = False
            self.wins = 0

            
            for eps in range(self.args.numEps):
                #if eps % 100 == 0:
                self.mcts = MCTS(self.game, self.nnet, self.args)
                self.episode = eps
                iterationTrainExamples += self.executeEpisode()
                eps_time.update(time.time() - end)
                end = time.time()
                bar.suffix = '({eps}/{maxeps} Eps Time: {et:.3f} | Total: {total:} | ETA: \
                                    {eta:}'.format(eps=eps+1, maxeps=self.args.numEps, et=eps_time.avg,
                                                   total=bar.elapsed_td, eta=bar.eta_td)
                bar.next()

            print("\n\nWins:", self.wins)
            self.writer.add_scalars('zero_10/winning', {'winning': self.wins}, i)
            self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                print("len(trainExamplesHistory) =", len(self.trainExamplesHistory),
                  " => remove the oldest trainExamples")
                self.trainExamplesHistory.pop(0)

            self.saveTrainExamples(i-1)
            trainExamples = []

            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

            self.nnet.train(trainExamples)

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint

        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration)+".examples")

        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
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

# Structure from: https://github.com/suragnair/alpha-zero-general/blob/master/Coach.py
