import random
from torch import Tensor
import numpy as np
import random
from torch import Tensor
from utils import *

class Game():
    
    def __init__(self, arena):
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
        
    
    def neuralNetworkInput(self, player, opponent):
        program_stack = self.integerImageRepresentation(player)
    
        for i in range(NUM_COMPETITORS):
            opponent_program = self.integerImageRepresentation(opponent[i])
            program_stack = np.concatenate([program_stack, opponent_program], axis=-1)
        
        return program_stack.reshape(NUM_COMPETITORS + 1, PROGRAM_WIDTH, PROGRAM_HEIGHT)
    
    def integerImageRepresentation(self, sequence):

        """ Creates the input for the neural network: """
        
        self.start = True
        self.x = (PROGRAM_WIDTH // 2) - (VOCAB_WIDTH // 2) 
        self.y = (PROGRAM_HEIGHT // 2) - (VOCAB_HEIGHT // 2) 
        self.program = np.full((PROGRAM_WIDTH, PROGRAM_HEIGHT), -1, dtype=np.int8)
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
        return np.array(binary).reshape((VOCAB_WIDTH, VOCAB_HEIGHT))
    
    def add_grid_to_program(self, grid):
        for x in range(VOCAB_WIDTH):
            for y in range(VOCAB_HEIGHT):
                self.program[self.x + x][self.y + y] = grid[x][y]
    
    def next_cord(self):
        
         #The program is initialized with -1, if it's something else we know its been filled already.\
         #The program is added in a spiral shape starting by moving to the right. Move down if left block \
         #is filled and bottom is emtpy, or move left if top is filled, or move up if right is filled, else \
         #move right.
        
        if self.start:
            self.x += VOCAB_WIDTH
            self.start = False
        elif self.x != 0 and self.program[self.x - 1, self.y] != -1 \
        and self.program[self.x, self.y + VOCAB_HEIGHT] == -1:
            self.y += VOCAB_HEIGHT
        elif self.y != 0 and self.program[self.x, self.y - 1] != -1:
            self.x -= VOCAB_WIDTH
        elif self.x + VOCAB_WIDTH != PROGRAM_WIDTH and self.program[self.x + VOCAB_WIDTH, self.y] != -1:
            self.y -= VOCAB_HEIGHT
        else:
            self.x += VOCAB_WIDTH
    
    def getInitProgram(self):
        
        return np.full((PREDICTION_LEN), -1, dtype=np.int8).tolist()
    
    def getGameEnded(self, playerOne, playerTwo):
        playerOne = self.integerImageRepresentation(playerOne)
        playerTwo = self.integerImageRepresentation(playerTwo)
        self.arena.add_players(playerOne, playerTwo)
        self.arena.run_steps(GAME_STEPS)
        reward, ones, twos = self.selectWinner(self.arena.grid().flatten())
        return reward, ones, twos, playerOne, playerTwo
    
    def selectWinner(self, game_result):
        ones = 0
        twos = 0
        
        for i in game_result:
            if i == 1:
                ones += 1
            elif i == 2:
                twos += 1

        if ones > twos:
            reward = 1.0
        elif twos > ones:
            reward = -1.0
        else:
            reward = 0.00000001
            
        return reward, ones, twos

    
    def stringRepresentation(self, program, opponent):

        string_state = ""
        for integer in program:
            string_state += str(integer)
        
        for i in range(NUM_COMPETITORS):
            for integer in opponent[i]:
                string_state += str(integer)
            
        return string_state
