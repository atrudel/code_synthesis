from GOLAI import arena
from settings import *

class Game():
    
    def __init__(self):
        
    
    
    def getInitBoard(self):
        return np.zeros((9,9), dtype=np.int8)
    
    def getBoardSize(self):
        return arena.size()
    
    def getActionSize(self):
        return(VOCABSIZE)
    
    def getNextState(self, board, player, action):
        
        """
        
        Turns the action/vocab into a board piece.
        Adds it to the board.
        Creates -1 for next action square.
        
        """
    
    def getGameEnded(self, board, player1, player2):
        
        player1, player2 = convert_to_arena_players(player1, player2)
        arena.add_players(player1, player2)
        arena.run_steps(GAMEROUNDS)
        
    def convert_to_arena_players(player1, player2):
        
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
        
        return board.reshape(BOARDWIDTH, BOARDHEIGHT)