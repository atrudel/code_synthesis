import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../../game/')))
from GOLAI.arena import Arena
from collections import deque
from MCTS import MCTS
import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time, os, sys
from pickle import Pickler, Unpickler
from random import shuffle
import random


class Coach():
    
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        #self.pnet = self.nnet.__class__(self.game)
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []
        self.allOpponents = []
        self.wins = 0
    
    def createRandomOpp(self):
        
        for i in range(self.args.numEps):
            value = list(range(0, self.args.vocabLen))  
            self.allOpponents.append(np.random.choice(value, self.args.predictionLen).tolist())
        
    
    def dirichlet_noise(self, probs):

        dim = (len(probs))
        probs = np.array(probs, dtype=float)
        new_probs = (1 - self.args.eps) * probs + self.args.eps * np.random.dirichlet(np.full(dim, self.args.alpha))

        return new_probs
    
    def executeEpisode(self, eps):
        
        trainExamples = []
        self.curProgram = self.game.getInitProgram()
        self.curOpponent = self.trainOpponents[eps]
        episodeStep = 0
        
        while True:
            episodeStep += 1
            temp = int(episodeStep < self.args.tempThreshold)
            
            pi = self.mcts.getActionProb(self.curProgram, self.curOpponent, temp)
            
            if episodeStep == 1 and self.args.dirichlet_noise:
                pi = self.dirichlet_noise(pi)
                
            trainExamples.append([self.game.integerImageRepresentation(self.curProgram), pi, None])
            action = np.random.choice(len(pi), p=pi)
            self.game.getNextState(self.curProgram, action)
                
            
            if episodeStep == self.args.predictionLen:
                self.allOpponents.append(self.curProgram)
                r = self.game.getGameEnded(self.curProgram, self.curOpponent)
                self.wins += r
                return[(x[0], x[1], r) for x in trainExamples]
            
    def learn(self):

        self.createRandomOpp()

        for i in range(1, self.args.numIters+1):

            print('---------Iter ' + str(i) + '---------')


            iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

            eps_time = AverageMeter()
            bar = Bar('Self Play', max=self.args.numEps)
            end = time.time()
            self.trainOpponents = self.allOpponents[-self.args.numEps:]
            shuffle(self.trainOpponents)
            self.wins = 0
            
            for eps in range(self.args.numEps):
                self.mcts = MCTS(self.game, self.nnet, self.args)
                iterationTrainExamples += self.executeEpisode(eps)
                eps_time.update(time.time() - end)
                end = time.time()
                bar.suffix = '({eps}/{maxeps} Eps Time: {et:.3f} | Total: {total:} | ETA: \
                {eta:}'.format(eps=eps+1, maxeps=self.args.numEps, et=eps_time.avg,\
                total=bar.elapsed_td, eta=bar.eta_td)

                bar.next()
            
            print("\n\nWins:", self.wins)
            
            self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                print("len(trainExamplesHistory) =", len(self.trainExamplesHistory), \
                      " => remove the oldest trainExamples")
                self.trainExamplesHistory.pop(0)

            #self.saveTrainExamples(i-1)
            trainExamples = []

            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')

            mcts = MCTS(self.game, self.nnet, self.args) ## why?
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
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
            
                
                
                
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    