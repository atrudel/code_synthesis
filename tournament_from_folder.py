import numpy as np
from game.game_of_life import gol
import os
from shutil import copyfile
import time
import sys
from lib.turn_program_into_file import turn_program_into_file
from game.game_of_life.rle import Pattern
from game.wrappers.GameContainer import GameContainer


if len(sys.argv) != 2:
    print(
"Error: Takes one directory as an argument. The directory should include player files with \n\
the .rle extension. Everyone plays against each other and get 3 points for a win and 1 point for a draw. \n\
Each score is added to the filename of each player and copied to a new folder under tournament-results. \n\n\
Example: python3 tournament_from_folder.py benchmark_players/test_players")
    exit(1)

dir_name = os.path.abspath(sys.argv[1]) + '/'
game = GameContainer(512, 512)
output_dir = "./tournament-results/"
run = int(time.time())
save_dir = output_dir + str(run) + '/'
os.makedirs(os.path.join(output_dir, str(run)))

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
    elif twos == ones:
        winner = 0
        
    return winner

def play_matches(players, dir_name, results):
    
    p1 = players[0]
    p2 = players[1]
    
    if p1 == 'Empty' or p2 == 'Empty':
        return results
    else:
        p1_prog = Pattern(dir_name + p1)
        p2_prog = Pattern(dir_name + p2)

        if len(p1_prog.data) != len(p2_prog.data):
            raise Exception("Players are not the same size!")

        p_size = int(len(p1_prog.data)**(0.5))

        p1_prog = np.reshape(np.array(p1_prog.data, dtype=np.int8), (p_size, p_size))
        p2_prog = np.reshape(np.array(p2_prog.data, dtype=np.int8), (p_size, p_size))

        game.add_players(p1_prog, p2_prog)
        game.launch(1000)
        game_ints = [x for x in game.gol.grid()]
        winner = decide_winner(game_ints)

        if winner == 1:
            results[p1] += 3
        elif winner == 2:
            results[p2] += 3
        else:
            results[p1] += 1
            results[p2] += 1
    
        return results

def match_programs(programs):
    if len(programs) % 2:
        programs.append('Empty')  

    rotation = list(programs)   

    matches = []
    for i in range(0, len(programs)-1):
        matches.append(rotation)
        rotation = [rotation[0]] + [rotation[-1]] + rotation[1:-1]
    
    matches = np.array(matches)
    len_size = matches.shape[0] * matches.shape[1]
    rounds = len_size // 2
    two_by_two = np.reshape(matches, (rounds, 2))  
    
    return two_by_two

programs = os.listdir(dir_name)
score = np.zeros(len(programs), dtype=np.int8)
results = dict(zip(programs, score))
list_of_games = match_programs(programs)
for i in (list_of_games):
    results = play_matches(i, dir_name, results)
for i in results:
    copyfile(dir_name + i, save_dir + str(results[i]) + "-" + i)

print("Done!")
