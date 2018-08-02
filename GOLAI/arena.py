import sys
from GOLEngine import GOLEngine
import numpy as np
from numpy import byte

class Arena:
    def __init__(self, width=512, height=512):
        self.board_dims = (height, width)
        self.engine = None
        self.player1 = None
        self.player2 = None
        self.steps = 0

    def setup(self):
        if self.player1 is not None:
            player1_padded = self.pad_vertically(self.player1)
        else:
            player1_padded = self.pad_vertically(np.zeros((1, 1), dtype=byte))

        if self.player2 is not None:
            # Rotate player2 180 degrees and replace 1's with 2's
            player2 = np.rot90(np.copy(self.player2), k=2) * 2
            player2_padded = self.pad_vertically(player2)
        else:
            player2_padded = self.pad_vertically(np.zeros((1, 1), dtype=byte))

        height, width = self.board_dims

        # Pad the two resulting blocks horizontally to form the final grid
        grid = self.pad_horizontally(player1_padded, player2_padded)
        self.engine = GOLEngine(width, height, grid)
        self.steps = 0

    def add_players(self, player1 = None, player2=None):
        ''' Fills the two player slots '''
        if player1 is not None:
            self.player1 = player1
        if player2 is not None:
            self.player2 = player2
        self.setup()

    def grid(self):
        if not self.engine:
            return np.zeros(self.board_dims)
        return self.engine.grid()

    def run_steps(self, num_steps):
        if self.player1 is None and self.player2 is None:
             raise Exception("This game doesn't have any players")

        try:
            self.engine.run_steps(num_steps)
            self.steps += num_steps
        except Exception as e:
            print(e)

    def size(self):
        return (self.board_dims[1], self.board_dims[0])

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
