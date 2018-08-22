
# coding: utf-8

# In[9]:


import numpy as np
import random
import pickle
from game.game_of_life import gol
from numba import jit
from multiprocessing import Pool
import os
import time


# In[10]:


from lib.turn_program_into_file import turn_program_into_file
from game.game_of_life.rle import Pattern
from game.wrappers.GameContainer import GameContainer


# In[13]:


Pattern('./benchmark_players/test_players/0.rle')
Pattern('./benchmark_players/spacefiller.rle')

