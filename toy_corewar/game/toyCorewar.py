import numpy as np
from numba import jit
import config

CFG = config.get_cfg().settings.toy_corewar

class ToyCorewar():
    """A toy implementation of Corewar with just 4 instructions"""
    
    def __init__(self, registers=None, num_registers=CFG.NUM_REGISTERS, mem_size=(4*CFG.MAX_LENGTH)):
        if registers is not None:
            self.registers = registers.clip(0, 255)
        else:
            self.registers = np.zeros(num_registers, dtype=int)
        self.memory = np.zeros(mem_size, dtype=int)
    
    @jit
    def load_player(self, player):
        i = 0
        for byte in player:
            if byte > 255:
                byte = 255
            if byte < 0:
                byte = 0
            self.memory[i] = byte
            i += 1
    
    def run(self, verbose=0):
        if verbose >= 2:
            print("Launching ToyCorewar!")
            self.print_state()
            print()
        pc = 0
        while pc is not None:
            pc = self.execute_instruction(pc, verbose)

    def execute_instruction(self, pc, verbose):
        """Execute the instruction located at pc, advance the pc and return it"""
        instructions = [self.no_instr, self.ld, self.st, self.add, self.sub]
        
        if pc <= len(self.memory) - 4:
            opcode = self.memory[pc]
            try:
                if opcode != 0:
                    instructions[opcode](self.memory[pc+1], self.memory[pc+2], self.memory[pc+3], verbose)
                    if verbose >= 2:
                        self.print_state()
                        print()
            except IndexError:
                if verbose:
                    print("Unknown opcode or register number")
            return pc + 4
        else:
             return None
    
    def no_instr(self, arg1, arg2, arg3, verbose):
        return
    
    def ld(self, arg1, arg2, arg3, verbose):
        """Load scalar value arg1 into register arg2"""
        if verbose > 0:
            print("Executing ld ", arg1, arg2)
        self.registers[arg2-1] = arg1
    
    def st(self, arg1, arg2, arg3, verbose):
        """Store value taken in register arg1 into register arg2"""
        if verbose > 0:
            print("Executing st ", arg1, arg2)
        self.registers[arg2-1] = self.registers[arg1-1]
    
    def add(self, arg1, arg2, arg3, verbose):
        if verbose > 0:
            print("Executing add", arg1, arg2, arg3)
        self.registers[arg3-1] = (self.registers[arg1-1] + self.registers[arg2-1]) % 256
    
    def sub(self, arg1, arg2, arg3, verbose):
        if verbose > 0:
            print("Executing sub", arg1, arg2, arg3)
        self.registers[arg3-1] = (self.registers[arg1-1] - self.registers[arg2-1]) % 256
    
    def print_state(self):
        print("Registers:")
        for reg in self.registers:
            print(repr(reg).rjust(2), end=" ")
        print()
        print("Memory:")
        for i, byte in enumerate(self.memory):
            print(repr(byte).rjust(2), end=" ")
            i += 1
            if i % 10 == 0:
                print()
