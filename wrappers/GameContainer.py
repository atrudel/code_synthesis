from gol import GameOfLife
import numpy as np

class GameContainer:
    def __init__(self, width=512, height=512):
        self.board_dims = (height, width)

    def add_players(self, player1=None, player2=None):
        ''' Fills the two player slots '''
        self.player1 = player1
        self.player2 = player2

    def launch(self, num_steps=0):
        ''' Creates the board, inserts the players' intial states into the grid
         and starts the game for num_steps '''

        if self.player1 is None:
             raise Exception("This game doesn't have any players")

        height, width = self.board_dims
        self.gol = GameOfLife(width, height)

        # Pad each of the players' initial states vertically
        player1_padded = self.pad_vertically(self.player1)
        if self.player2 is not None:
            # Rotate player2 180 degrees and replace 1's with 2's
            player2 = np.rot90(np.copy(self.player2), k=2) * 2
            player2_padded = self.pad_vertically(player2)
        else:
            dummy = np.zeros(self.player1.shape, dtype=np.int8)
            player2_padded = self.pad_vertically(dummy)

        # Pad the two resulting blocks horizontally to form the final grid
        grid = self.pad_horizontally(player1_padded, player2_padded)
        test = grid.flatten()
        u, indices = np.unique(test, return_index=True)
        
        self.gol.set_grid(bytearray(grid))
        self.run_steps(num_steps)

    def run_steps(self, num_steps):
        try:
            self.gol.run_steps(num_steps)
        except AttributeError:
            print("gol object wasn't initialized. launch() before you run_steps()")

    def pad_vertically(self, state):
        ''' Adds equal zero padding to the top and bottom of a player's initial state
        in order to match the board's height. Handles off-by-one padding distance '''

        total_height = self.board_dims[0]
        total_width = state.shape[1]
        top = (total_height - state.shape[0]) // 2
        bottom = total_height - top - state.shape[0]

        top_padding = np.zeros((top, total_width), dtype=np.int8)
        bottom_padding = np.zeros((bottom, total_width), dtype=np.int8)
        return np.vstack((top_padding, state, bottom_padding))

    def pad_horizontally(self, player1, player2):
        ''' Adds zero padding to position players uniformly away from each other
        and from the edges of the game on the horizontal axis '''

        total_height, total_width = self.board_dims
        extremity = (total_width - player1.shape[1] - player2.shape[1]) // 3
        middle = total_width - player1.shape[1] - player2.shape[1] - 2 * extremity

        extremity_padding = np.zeros((total_height, extremity), dtype=np.int8)
        middle_padding = np.zeros((total_height, middle), dtype=np.int8)
        return np.hstack((extremity_padding, player1, middle_padding,
                        player2, extremity_padding))
