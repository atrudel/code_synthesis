# Code Synthesis with Reinforcement Learning

This is an implementation of Deep reinforcement learning for code synthesis loosely inspired from the paper [Code Syhthesis through Reinforcement Learning Guided Tree Search](https://arxiv.org/pdf/1806.02932.pdf). We wanted to to create a RL agent that would generate a sequence of code instructions when given a specific task to accomplish.

For this purpose, we created ToyCorewar, a small framework that executes assembly-like instructions.
Toy Corewar has 4 registers, whose values can range from 0 to 255.

### Toy Corewar instructions
- ld: load a value (0-19) in a register
- st: store a value from a register to another
- add: add the values of two registers and store the sum in another one
- sub: subtract the values of two registers and store the difference in another one

### Task
All 4 registers are initialized to 0, and the task is to set them to their target values within the constraints of the set of instructions.
At each timestep, the current and target register values are communicated to the agent by the environment.

### Reinforcement Learning environment
- For code synthesis, each action represents the choice of an instruction and its operands (255 possible actions)
- Two types of agents were implemented (DQN, Actor-critic).
- An Environment class is implemented, which executes the code in a ToyCorewar VM and returns appropriate reward.

## Requirements
- python 3.6.5
- numpy
- torch
- tensorboardX
- sigopt

`pip install -r requirements.txt`

## Usage
How to launch a training:
- Specify the configuration in **config.json**
- Specify the training procedure in **training.json**
- Launch the training

`python main.py [name_of_experiment]`
- Check the logs in the folder **Experiment**

![alt text](https://github.com/atrudel/code_synthesis/blob/master/visuals/log_example.png "Log example")

- Visualize the training graph

`tensorboard --logdir runs`

![alt text](https://github.com/atrudel/code_synthesis/blob/master/visuals/graph_example.png "Graph example")

## Hyperparameter optimization
- Create an account on sigopt.com
- Change API token at line 75 in **parameter_search.py**
- Specify training task at line 13 in **parameter_search.py**
- Launch parameter search

`python parameter_search.py [name_of_experiment]`
