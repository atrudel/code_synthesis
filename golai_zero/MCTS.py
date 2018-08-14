import math
import numpy as np
from copy import deepcopy
EPS = 1e-8

class MCTS():
    
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # Q values for state, actions 
        self.Nsa = {}  # Edge state, action was visited
        self.Ns = {}   # Board state was visited
        self.Ps = {}   # Policy from neural network
        self.Es = {}   # Stores the rewards from game.getGameEnded
    
    def getActionProb(self, program, temp=1):
        
        for i in range(self.args.numMCTSSims):
            program_sim = deepcopy(program)
            self.search(program_sim)
            
        s = self.game.stringRepresentation(program)
        counts = [self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in range(self.args.vocabLen)]
        
        if temp == 0:
            bestAction = np.argmax(counts)
            probs = [0]*len(counts)
            probs[bestAction]=1
            return probs
        
        counts = [x**(1./temp) for x in counts]
        probs = [x/float(sum(counts)) for x in counts]
        
        return probs
    
    def search(self, program):
        
        s = self.game.stringRepresentation(program)
        
        if program[-1] != -1:
            _, v = self.nnet.predict(self.game.integerImageRepresentation(program))
            return v
                
        if s not in self.Ps:
            self.Ps[s], v = self.nnet.predict(self.game.integerImageRepresentation(program))
            self.Ns[s] = 0
            return v
        
        cur_best = -float('inf')
        best_act = -1
        
        for a in range(self.args.vocabLen):
            if (s,a) in self.Qsa:
                u = self.Qsa[(s,a)] + self.args.cpuct*self.Ps[s][a]*math.sqrt(self.Ns[s])/(1+self.Nsa[(s,a)])
            else:
                u = self.args.cpuct*self.Ps[s][a]*math.sqrt(self.Ns[s] + EPS)
            if u > cur_best:
                cur_best = u
                best_act = a
        
        a = best_act
        next_program = self.game.getNextState(program, a)
        v = self.search(next_program)
        
        if (s, a) in self.Qsa:
            self.Qsa[(s,a)] = (self.Nsa[(s,a)]*self.Qsa[(s,a)] + v)/(self.Nsa[(s,a)]+1)
            self.Nsa[(s,a)] += 1
        else:
            self.Qsa[(s,a)] = v
            self.Nsa[(s,a)] = 1
        
        self.Ns[s] += 1
        return v
    
# Structure from: https://github.com/suragnair/alpha-zero-general/blob/master/MCTS.py