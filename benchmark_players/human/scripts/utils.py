import sys
sys.path.insert(0, '/Users/ademenet/Github/GOLAI/game/game_of_life')
sys.path.insert(0, '/Users/ademenet/Github/GOLAI/benchmarks/lib')
import os
import matplotlib.pyplot as plt
import numpy as np
import rle
from turn_program_into_file import turn_program_into_file

def import_rle(file):
    """Open `.rle` file and return a numpy array."""
    pattern = rle.Pattern(file)
    return pattern.data

def viz_program(program_array, offsets, ref_shape=(64,64)):
    """Take a numpy array and display it.

    Arguments:
        program_array (numpy array): the program you wan't to visualize. Could be smaller than (64, 64).
        offsets (tuple): define the offsets coordinates.
        ref_shape (tuple): we assume that the reference shape is (64, 64).
    """
    if program_array.shape is not ref_shape:
        padded_program = np.zeros(ref_shape, dtype=int)
        insert_offset(padded_program, program_array, offsets)
    plt.imshow(padded_program)

def offset_centered(array_shape, ref_shape=(64,64)):
    """Return centered on ref_shape coordinates."""
    y = (ref_shape[0] - array_shape[0]) / 2
    x = (ref_shape[1] - array_shape[1]) / 2
    return (int(y), int(x))

def insert_offset(big_array, small_array, offsets):
    """Insert small array into big array at the given offsets (assuming it is representing the upper-left points).

    Arguments:
        big_array (numpy array): must be bigger than small_array.
        small_array (numpy array): must be smaller than big_array.
        offsets (tuple): must be the same dimensions as big_array and small_array.

    Return:
        Numpy array of big_array's dimensions.
    """
    insert_here = [slice(offsets[dim], offsets[dim] + small_array.shape[dim]) for dim in range(small_array.ndim)]
    big_array[insert_here] = small_array
    return(big_array)

def tiling_pattern(pattern, ref_shape=(64,64), padding=(3,3)):
    """Tile the pattern (numpy array) as much as possible and return ref_shape size numpy array."""
    size_x = pattern.shape[1]
    size_y = pattern.shape[0]
    offset_x = 0 + padding[1]
    result = np.zeros(ref_shape, dtype=int)
    while offset_x + size_x <= 64:
        offset_y = 0 + padding[0]
        while offset_y + size_y <= 64:
            insert_offset(result, pattern, (offset_y, offset_x))
            offset_y += size_y + padding[0]
        offset_x += size_x + padding[1]
    return result

def save_to_file(program, filename="program.rle", name="", author="", comments=""):
    """Save numpy array to a `.rle` file."""
    turn_program_into_file(program, filename, name, author, comments)
