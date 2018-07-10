# Game of Life 2 Player

### Game of life core rules
- Any live cell with fewer than two live neighbors dies, as if by under population.
- Any live cell with two or three live neighbors lives on to the next generation.
- Any live cell with more than three live neighbors dies, as if by overpopulation.
- Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction.

### Game of life 2 Player rules
- The game board is 512 x 512
- Each player gets a 64x64 grid to develop their program 
- The player that has the most surrounding cells, owns the new cell that is born
- Killing cells follows them Game of life core rules
- The player with most live cells after 1000 cycles wins

## Requirements
- python 3.6.5 (is guaranteed to work)
- numpy
- numba
- pyqt5 (for the graphical interface)

`pip install numpy numba pyqt5`
