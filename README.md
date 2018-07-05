## Benchmark 1 -  Random King of the Hill 
**How:** Two random players are created. The winner stays on the board and is challenged by a new random player.  
**Reasoning:** This benchmark can indicate if an AI generated program is improving faster than chance. 

## Benchmark 2 -  Famous GOL Patterns 
**How:** Classic GOL patterns adopted as players.  
**Reasoning:** Although these patterns are not designed as players, they can indicate how well an AI generated program interacts with a growing program.   

## Benchmark 3 - Classic generative algorithm 
**How:**  
**Reasoning:** 


## Usage:

### Turn an numpy array into a .rle file
```python
from lib.turn_program_into_file import turn_program_into_file
turn_program_into_file(program, filename="program.rle", name="", author="", comments=""):

# program: should be a numpy array with the np.int8 format
# filname: should include the name and path where you want the file to be saved
# name: is the name of the experiment
# author: is the name of the author
# comments: additional context
``` 
[Here are more details about the RLE format.](http://www.conwaylife.com/wiki/RLE)
 
### Create a Round-robin tournament(everyone plays everyone) with all players in a folder
```bash
python3 tournament_from_folder.py benchmark_players/test_players

# Takes one directory as an argument. The directory should include player files with  
# the .rle extension. Everyone plays against each other and get 3 points for a win and 1 point for a draw.  
# Each score is added to the filename of each player and copied to a new folder under tournament-results.  
```
