import GOLAI.gameView as gw
from GOLAI.arena import Arena
from GOLAI.fileLoader import Pattern, pad_pattern
import numpy as np

game = Arena(27, 27)
p1 = Pattern("/Users/ewallner/Desktop/output/1534777080/90/19-49-lost-tiles-0-p1.rle")
p2 = Pattern("/Users/ewallner/Desktop/output/1534777080/90/19-49-win-tiles-62-p2.rle")

playerOne = pad_pattern(p1.data, (6,6))
playerTwo = pad_pattern(p2.data, (6,6))
game.add_players(np.array(playerOne), np.array(playerTwo))

if __name__ == '__main__':
        gw.start(game)
