# Toy Corewar

A small framework that executes assembly-like instructions for code synthesis.
Toy Corewar has 4 registers, whose values can range from 0 to 255.

### Toy Corewar instructions
- ld: load a value (0-19) in a register
- st: store a value from a register to another
- add: add the values of two registers and store the sum in another one
- sub: subtract the values of two registers and store the difference in another one

### Reinforcement Learning environment
- For code synthesis, each action represents the choice of an instruction and its operands.
- Three types of agents are implemented (DQN, Actor-critic, Policy Gradients).
- An Environment class is implemented, which executes the code in a ToyCorewar VM and returns appropriate reward.

## Requirements
- python 3.6.5
- numpy
- torch
- tensorboardX
- sigopt

`install -r requirements.txt`

## Usage
Hot to launch a training:
- Specify the configuration in **config.json**
- Specify the training procedure in **training.json**
- Launch the training

`python main.py [name_of_experiment]`
- Check the logs in the folder **Experiment**
- Visualize the training graph

`tensorboard --logdir runs`

## Hyperparameter optimization
- Create an account on sigopt.com
- Change API token at line 75 in **parameter_search.py**
- Specify training task at line 13 in **parameter_search.py**
- Launch parameter search

`python parameter_search.py [name_of_experiment]`
