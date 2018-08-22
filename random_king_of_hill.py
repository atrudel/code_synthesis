
# coding: utf-8

# In[1]:


import numpy as np
import random
import pickle
from game.game_of_life import gol
from numba import jit
from multiprocessing import Pool
import os
import time


# In[2]:


from lib.turn_program_into_file import turn_program_into_file
from game.wrappers.GameContainer import GameContainer


# In[3]:


cpus = 4
player_size = 64
game = GameContainer(512, 512)
# Each run taks 2.6 sec per cpu, 38089 = a 24h run.
run_limit = 10
output_dir = "./output/KOH/"
run = int(time.time())
save_dir = output_dir + str(run) + '/'


# In[4]:


king_hill_ids = list(range(0, cpus))
os.makedirs(os.path.join(output_dir, str(run)))


# In[5]:


def get_players(player_size):
    
    value = [1, 0]  
    dist = random.random()
    p = np.random.choice(value, (player_size, player_size), p=[dist, (1.0 - dist)])
    
    return np.array(p, dtype=np.int8)


# In[6]:


def decide_winner(game_result):
    ones = 0
    twos = 0
    winner = 1
    
    for i in game_result:
        if i == 1:
            ones += 1
        elif i == 2:
            twos += 1
            
    if twos > ones:
        winner = 2
    return winner


# In[7]:


def run_king_of_hill(list_id):
    
    p1 = get_players(player_size)
    p2 = get_players(player_size)
    start_time = time.time()

    for i in range(run_limit):
        print(list_id)
        print("--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        game.add_players(p1, p2)
        game.launch(1000)
        game_ints = [x for x in game.gol.grid()]
        winner = decide_winner(game_ints)

        if winner == 1:
            p2 = get_players(player_size)
        else:
            p1 = get_players(player_size)


    if winner == 1:
        program = p1
    else:
        program = p2
    
    turn_program_into_file(program, save_dir + str(list_id) + ".rle", "RKH", "EW", "")


# In[8]:


if __name__ == "__main__":

    pool = Pool(processes=cpus) 
    data_all = pool.map(run_king_of_hill, king_hill_ids)
    print("Done!") 

