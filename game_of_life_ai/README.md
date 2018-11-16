## Game of Life Two-player
A reinforcement learning model for the Two-player Game of Life based on Deepmind's AlphaGo Zero.

### Rules
- Board size: the size of the board that the players play on
- Player size: the number of initial tiles that each player starts with
- Cycles: the number of cycles to run before you decide who wins
- Winner: the player with the most amount of tiles at the end wins

### Reinforcement Learning environment
- The agent is made out of three neural networks: a feature extractor, a value network, and a policy network. 
- The model uses Monte Carlo Tree Search to simulate future actions.
- The environment is the Game of life game found in the main repository.

## Requirements
- python 3.6.5
- numpy
- torch
- tensorboardX

## Usage
Hot to launch a training:
- Specify the configuration in **utils.py**
- Launch the training

`python3 multi_main.py`
- Check the logs in the folder **output**
- Visualize the training graph

`tensorboard --logdir runs`
