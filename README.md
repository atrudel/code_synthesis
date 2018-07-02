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
- If both players have an equal amount of surrounding cells, each player has a random 50/50 chance of owning the new cell
- Killing cells follows them Game of life core rules
- The player that has the most living cells after 1000 cycles wins the game
- If both players have an equal amount of cells it is a draw

## Getting Started Directory
Under the getting started dir, we'll develop a minimal viable game to start developing players.

### Todo
- [ ] The main game of life game
- [ ] Adapt the main game for two players
- [ ] Create a wrapper to use in deep learning models


## Core Development Directory

Here, we'll develop an optimized version 
