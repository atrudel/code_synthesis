from config import *
import numpy as np
from game.toyCorewar import ToyCorewar
from config import *

# Specific values for all registers, without division
def specific_register_values(cw):
    target_values = np.array([0,10,10,20], dtype=int)
    reward = 0
    for reg in range(N_TARGETS):
        reward -= abs(target_values[reg] - cw.registers[reg])
    return cw.registers, target_values, reward

# Specific values for all registers, with division
def specific_register_values_division(cw):
    target_values = np.array([0,10,10,20], dtype=int)
    reward = 0
    for reg in range(N_TARGETS):
        reward -= abs((target_values[reg] - cw.registers[reg]) / (target_values[reg] + 1))
    return cw.registers, target_values, reward

# Specific value for one register
def one_register_value(cw):
    target = 55
    register = 3 # Reminder: register indexes start from 1
    target_values = np.zeros(4, dtype=int)
    target_values[register-1] = target
    reward = -abs(target - cw.registers[register-1])
    return cw.registers, target_values, reward

# Maximize the sum of all register values
def maximize_all_registers(cw):
    target = np.zeros(4, dtype=int)
    reward = cw.registers.sum()
    return cw.registers, target, reward

# Minimize the sum of all register values
def minimize_all_registers(cw):
    target = np.zeros(4, dtype=int)
    reward = -cw.registers.sum()
    return cw.registers, target, reward

def maximize_one_register(cw):
    register = 4
    target = np.zeros(4, dtype=int)
    reward = cw.registers[register-1]
    return cw.registers, target, reward

def minimize_one_register(cw):
    register = 2
    target = np.zeros(4, dtype=int)
    reward = -cw.registers[register-1]
    return cw.registers, target, reward

reward_functions = [specific_register_values,
                   specific_register_values_division,
                   one_register_value,
                   maximize_all_registers,
                   minimize_all_registers,
                   maximize_one_register,
                   minimize_one_register]