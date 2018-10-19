import config
from game.toyCorewar import ToyCorewar
from game.program_synthesis import Program, Instruction
import numpy as np
import torch
import math

CFG = config.get_cfg()
CWCFG = CFG.settings.toy_corewar

class Env():
    action_space_n = CWCFG.NUM_ACTIONS
    
    def __init__(self, reward_func):
        '''The reward function must take a ToyCorewar instance as an argument and return a reward'''
        self.reward_func = reward_func
        self.best_score = -math.inf
    
    def step(self, action):
        '''action is a number between 0 and action_space_n'''
        # Check if episode is terminated
        assert not self.done, "Can't add an action to a finished episode. Call Env.reset()"
        
        # Convert action to a program instruction
        opcode, arg1, arg2, arg3 = Env.interpret_action(action) if isinstance(action, int) else action
        
        # Handle the no-action case
        if opcode is None:
            done = True
        else:
            # Add instruction to the program
            done = self.program.add_instruction(Instruction(opcode, arg1, arg2, arg3))
        
        # Load the new program in a new toycorewar and run it
        cw = ToyCorewar()
        cw.load_player(self.program.to_byte_sequence())
        cw.run()
        
        # Create and format the state
        state = self.build_state(cw)
        
        # Calculate reward
        _,_, reward = self.reward_func(cw) #if opcode is not None else (None, None, 0)
        self.done = done
        
        # Reward is only given on the final state
        if self.done:
            self.total_reward += reward
            self.best_score = max(self.total_reward, self.best_score)
        else:
            reward = 0
        
        return state, reward, done, None
        
    def reset(self, reg_init=None):
        # Reset program and return empty state
        cw = ToyCorewar(reg_init=reg_init)
        self.program = Program()
        self.total_reward = 0
        self.done = False
        return self.build_state(cw)
    
    def build_state(self, cw):
        # Take self.program and self.cw to build a representation of the state
        program_state = self.program.to_embedding_sequence()
        current_mem, target_mem, _ = self.reward_func(cw)
        memory_state = np.array([current_mem, target_mem])
        return program_state, memory_state
        
    def interpret_action(action):
        # ld
        if action < 80:
            opcode = 1
            div, reminder = divmod(action, 20)
            arg1 = reminder
            arg2 = div + 1
            arg3 = None
        # st
        elif action < 96:
            opcode = 2
            div, reminder = divmod(action - 80, 4)
            arg1 = reminder + 1
            arg2 = div + 1
            arg3 = None
        # add
        elif action < 160:
            opcode = 3
            div, reminder = divmod(action - 96, 4)
            arg1 = reminder + 1
            div, reminder = divmod(div, 4)
            arg2 = reminder + 1
            arg3 = div + 1
        # sub
        elif action < 224:
            opcode = 4
            div, reminder = divmod(action - 160, 4)
            arg1 = reminder + 1
            div, reminder = divmod(div, 4)
            arg2 = reminder + 1
            arg3 = div + 1
        # end (no instruction)
        else:
            return (None, None, None, None)
        return opcode, arg1, arg2, arg3

    def print_details(self, file=None):
        for i, subprogram in enumerate(self.program):
            cw = ToyCorewar()
            cw.load_player(subprogram.to_byte_sequence())
            cw.run()
            _, _, reward = self.reward_func(cw)
            subprogram[i].print(file=file, end=' ')
            details = "\t[ "
            for reg in range(len(cw.registers)):
                details += str(cw.registers[reg]).rjust(3) + " "
            details += "]  "
            details += str(reward).rjust(4)
            print(details, file=file)
        print("-" * 37, file=file)
        print("Total reward: {}".format(self.total_reward).rjust(37), file=file)
