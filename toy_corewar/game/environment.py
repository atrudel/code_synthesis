import config
from game.toyCorewar import ToyCorewar
from game.program_synthesis import Program, Instruction
import numpy as np
import torch
import math

CFG = config.get_cfg()
CWCFG = CFG.settings.toy_corewar

class Env():
    opcode_comb = [
                    CWCFG.NUM_REGISTERS * CWCFG.N_VALS,
                    CWCFG.NUM_REGISTERS * CWCFG.N_VALS + CWCFG.NUM_REGISTERS ** 2,
                    CWCFG.NUM_REGISTERS * CWCFG.N_VALS + CWCFG.NUM_REGISTERS ** 2 + CWCFG.NUM_REGISTERS ** 3,
                    CWCFG.NUM_REGISTERS * CWCFG.N_VALS + CWCFG.NUM_REGISTERS ** 2 + 2 * (CWCFG.NUM_REGISTERS ** 3)
                    ]

    action_space_n = CWCFG.NUM_ACTIONS
    
    def __init__(self, reward_func):
        '''reward_func is an instantiated Reward_function object'''
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
            self.done = True
            if len(self.program) == 0:
                self.register_states.append(self.register_states[0])
            reward = self.reward_func(self)

        else:
            # Add instruction to the program
            self.done = self.program.add_instruction(Instruction(opcode, arg1, arg2, arg3))
            cw = ToyCorewar()
            cw.load_player(self.program.to_byte_sequence())
            cw.run()
            self.register_states.append(cw.reg_state())
            reward = self.reward_func(self)
            self.rewards.append(reward)
            self.total_reward += reward

        # Record best
        if self.done:
            self.performance = self.reward_func.performance(self)
            self.best_score = max(self.performance, self.best_score)

        # Create and format the state
        state = self.build_state()

        return state, reward, self.done, None


    def reset(self, reg_init=None):
        # Reset program and return empty state
        self.reg_init = reg_init if reg_init is not None else np.zeros(CWCFG.NUM_REGISTERS, dtype=int)
        self.program = Program()
        self.rewards = []
        self.total_reward = 0
        self.performance = None
        self.register_states = [self.reg_init]
        self.done = False
        return self.build_state()
    
    def build_state(self):
        # Take self.program and self.cw to build a representation of the state
        program_state = self.program.to_embedding_sequence()
        current_mem = self.register_states[-1]
        target_mem = self.reward_func.targets
        memory_state = np.array([current_mem, target_mem])
        return program_state, memory_state
        
    def interpret_action(action):
        regs = CWCFG.NUM_REGISTERS
        # ld
        if action < Env.opcode_comb[0]:
            opcode = 1
            div, reminder = divmod(action, CWCFG.N_VALS)
            arg1 = reminder
            arg2 = div + 1
            arg3 = None
        # st
        elif action < Env.opcode_comb[1]:
            opcode = 2
            div, reminder = divmod(action - Env.opcode_comb[0], regs)
            arg1 = reminder + 1
            arg2 = div + 1
            arg3 = None
        # add
        elif action < Env.opcode_comb[2]:
            opcode = 3
            div, reminder = divmod(action - Env.opcode_comb[1], regs)
            arg1 = reminder + 1
            div, reminder = divmod(div, regs)
            arg2 = reminder + 1
            arg3 = div + 1
        # sub
        elif action < Env.opcode_comb[3]:
            opcode = 4
            div, reminder = divmod(action - Env.opcode_comb[2], regs)
            arg1 = reminder + 1
            div, reminder = divmod(div, regs)
            arg2 = reminder + 1
            arg3 = div + 1
        # end (no instruction)
        else:
            return (None, None, None, None)
        return opcode, arg1, arg2, arg3


    def print_details(self, file=None):
        for i, instruction in enumerate(self.program):
            self.program[i].print(file=file, end=' ')
            details = "\t[ "
            for reg in range(CWCFG.NUM_REGISTERS):
                details += str(self.register_states[i + 1][reg]).rjust(3) + " "
            details += "]  "
            details += str(self.rewards[i]).rjust(4)
            print(details, file=file)
        print("-" * 37, file=file)
        print("Total reward: {}".format(self.total_reward).rjust(37), file=file)
        print("Performance: {}".format(self.performance).rjust(37), file=file)
