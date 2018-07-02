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
- Score
  - Each game is played in two rounds until cycle 1000, the players switch position after round 1
  - For each round, each player gets their living cells minus their opponents cells
  - The score is calculated at cycle 1000
  - Each player gets a total score by adding their score for each round
  - The player with the highest total score wins
    - If they have the same total score, the player with the highest round score wins
    - If they have the same highest score, a player is randomly picked as a winner

### Input/Output of the Game
- The game receives two 64x64 programs and are placed in the initial game
- The game runs for 1000 cycles 
- The game outputs the number of live cells for each player

## Getting Started Directory
Under the getting started dir, we'll develop a minimal viable game to start developing players.

### Todo
- [ ] The main game of life game
- [ ] Adapt the main game for two players

## Core Development Directory
Here, we'll develop an optimized version 
- [ ] The main game of life game
- [ ] Adapt the main game for two players
- [ ] A GUI that looks like the original game of life with one color for each player
- [ ] Be able to jump to a specific cycle or go step by step
