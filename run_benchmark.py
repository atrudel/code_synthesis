import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../game')))
from lib.turn_program_into_file import turn_program_into_file
from GOLAI.arena import Arena
from tensorboardX import SummaryWriter
from GOLAI.fileLoader import Pattern, pad_pattern
import random
import re

import numpy as np
from shutil import copyfile
import time

if len(sys.argv) != 2:
    exit(1)
    
benchmark_dir = (sys.argv[1])

arena = Arena(26, 26)
output_dir = "./output/tournament-results/"
run = int(time.time())
save_dir = output_dir + str(run) + '/'
os.makedirs(os.path.join(output_dir, str(run)))
writer = SummaryWriter()

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]


game_intervals = os.listdir(benchmark_dir)
game_intervals.sort(key=natural_keys)

def file_to_program(path):
    player = Pattern(path)
    player = pad_pattern(player.data, (6,6))
    return(np.array(player))

def turn_folder_into_player(folder):
    #print(folder)
    programs = []
    for program_file in os.listdir(folder):
        if program_file[-5:] == '1.rle':
            print(os.path.join(folder, program_file))
            programs.append(file_to_program(os.path.join(folder, program_file)))
    return programs

def calc_winner(game_result):
    ones = 0
    twos = 0

    for i in game_result:
        if i == 1:
            ones += 1
        elif i == 2:
            twos += 1

    if ones > twos:
        reward = 1
    elif twos > ones:
        reward = -1
    else:
        reward = 0

    return reward


latest_programs = turn_folder_into_player(os.path.join(benchmark_dir, game_intervals[-1] + '/'))

def run_games(games):
    score = 0
    for game in games:
        arena.add_players(game[0], game[1])
        arena.run_steps(100)
        score += calc_winner(arena.grid().flatten())
    return score

game_round = 5
for folder in game_intervals:
    competitors = turn_folder_into_player(os.path.join(benchmark_dir,folder))
    final_score = 0
    for i in range(5):
        random.shuffle(competitors)
        games = list(zip(latest_programs, competitors))
        final_score += run_games(games)
    final_score /= 5
    writer.add_scalars('Benchmark/winning', {'winning': final_score}, game_round)
    game_round += 5
    
print("Benchmark complete!")