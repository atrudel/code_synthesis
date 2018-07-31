from GOLAI import arena
import random
from settings import *
from torch import Tensor

class Game():
    
    def __init__(self):
        
    
    def getInitBoard(self):
        return np.zeros((9,9), dtype=np.int8)
    
    def getBoardSize(self):
        return arena.size()
    
    def getActionSize(self):
        return(VOCAB_SIZE)
    
    def getNextState(self, program, action):
        
        """
        
        Turns the action/vocab into a board piece.
        Adds it to the board.
        Creates -1 for next action square.
        
        """
    
    def getGameEnded(self, playerOne, playerTwo, game_round):
        
        if game_round < PROGRAM_LENGHT:
            return 0
        else:
            player1, player2 = convertToArenaPlayers(playerOne, playerTwo)
            arena.add_players(playerOne, playerTwo)
            arena.run_steps(GAME_ROUNDS)
            return selectWinner(arena.grid())
    
    def selectWinner(self, board):
        ones = 0, twos = 0

        for i in game_result:
            if i == 1:
                ones += 1
            elif i == 2:
                twos += 1

        if ones > twos:
            winner = Tensor(1.0)
        elif ones > twos:
            winner = Tensor(-1.0)
        else:
            winner = Tensor(random.uniform(0.001, 0.1))
            
        return winner
        
    def convertToArenaPlayers(player1, player2):
        
        """
        
        Turns the working format into a 2D numpy array int8
        
        """
        
        return player1, player2
    
    def stringRepresentation(self, board):
        
        """
        
        Turn board into a string used for the MCST hashing.
        
        """
        
        board_string = ""
        for integer in board:
            board_string += str(integer)
            
        return board_string
    
    def integerImageRepresentation(self, board):
        
        """
        
        Creates the input for the neural network:
        
        """
        
        return board.reshape(PROGRAM_WIDTH, PROGRAM_HEIGHT)
    

# Structure from: https://github.com/suragnair/alpha-zero-general/blob/master/Game.py