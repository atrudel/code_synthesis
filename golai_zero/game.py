import random
from torch import Tensor
import numpy as np

import random
from torch import Tensor

class Game():
    
    def __init__(self, args, arena):
        self.args = args
        self.arena = arena
        
    def getNextState(self, program, action):
        
        """
        
        Turns the action/vocab into a board piece.
        Adds it to the board.
        Creates -1 for next action square.
        
        """
        for i in range(len(program)):
            if program[i] == -1:
                program[i] = action
                return program
        print("getNextState was called with a full program")
        return program
        
    def integerImageRepresentation(self, sequence):

        """ Creates the input for the neural network: """
        
        self.start = True
        self.x = (self.args.programWidth // 2) - (self.args.vocabWidth // 2) 
        self.y = (self.args.programHeight // 2) - (self.args.vocabHeight // 2) 
        self.program = np.full((self.args.programWidth, self.args.programHeight), -1, dtype=np.int8)
        self.create_player_from_sequence(sequence)
        return self.program
    
    def create_player_from_sequence(self, sequence):
        for digit in sequence:
            if digit == -1:
                break
            grid = self.digit_to_grid(digit)
            self.add_grid_to_program(grid)
            self.next_cord()
                
    def digit_to_grid(self, digit):
        # Turn digit into binary representation to create a word
        binary = "{0:b}".format(digit)
        binary = binary.zfill(4)
        binary = list(str(binary))
        return np.array(binary).reshape((self.args.vocabWidth, self.args.vocabHeight))
    
    def add_grid_to_program(self, grid):
        for x in range(self.args.vocabWidth):
            for y in range(self.args.vocabHeight):
                self.program[self.x + x][self.y + y] = grid[x][y]
    
    def next_cord(self):
        
         #The program is initialized with -1, if it's something else we know its been filled already.\
         #The program is added in a spiral shape starting by moving to the right. Move down if left block \
         #is filled and bottom is emtpy, or move left if top is filled, or move up if right is filled, else \
         #move right.
        
        if self.start:
            self.x += self.args.vocabWidth
            self.start = False
        elif self.x != 0 and self.program[self.x - 1, self.y] != -1 \
        and self.program[self.x, self.y + self.args.vocabHeight] == -1:
            self.y += self.args.vocabHeight
        elif self.y != 0 and self.program[self.x, self.y - 1] != -1:
            self.x -= self.args.vocabWidth
        elif self.x + self.args.vocabWidth != self.args.programWidth and self.program[self.x + self.args.vocabWidth, self.y] != -1:
            self.y -= self.args.vocabHeight
        else:
            self.x += self.args.vocabWidth
    
    def getInitProgram(self):
        
        return np.full((self.args.predictionLen), -1, dtype=np.int8).tolist()
    
    def getGameEnded(self, playerOne, playerTwo):
        playerOne = self.integerImageRepresentation(playerOne)
        playerTwo = self.integerImageRepresentation(playerTwo)
        self.arena.add_players(playerOne, playerTwo)
        self.arena.run_steps(self.args.gameSteps)
        return self.selectWinner(self.arena.grid().flatten())
    
    def selectWinner(self, game_result):
        ones = 0
        twos = 0
        
        for i in game_result:
            if i == 1:
                ones += 1
            elif i == 2:
                twos += 1

        if ones > twos:
            winner = 1.0
        elif ones > twos:
            winner = -1.0
        else:
            winner = 0.00000001
            
        return winner

    
    def stringRepresentation(self, program):

        program_string = ""
        for integer in program:
            program_string += str(integer)
            
        return program_string