class Player():
    def __init__(self):
        self.start = True
        self.vocab = 512
        self.program_size = 33
        self.vocab_w = 3
        self.vocab_h = 3
        self.program = np.full((self.program_size, self.program_size), -1)
        self.x = ((self.program_size - self.vocab_w) // 2) - 1
        self.y = ((self.program_size - self.vocab_h) // 2) - 1
        self.prediction_len = self.program_size // self.vocab_wh
        
    
    def create_player_from_sequence(self, vocab_sequence):  
        for word in vocab_sequence:
            grid = word_to_grid(self, word)
            add_grid_to_program(self, grid)
            next_cord(self)
            
            
    def word_to_grid(self, word):
        # Turn digit into binary representation to create a word
        binary = "{0:b}".format(word)
        binary = binary.zfill(9)
        return np.array(binary).reshape((self.vocab_w, self.vocab_h))
    
    def add_grid_to_program(self, grid):
        for y in self.vocab_h:
            for x in self.vocab_wh:
                self.program[self.x + x][self.y + y] = grid[x][y]
    
    def next_cord(self):
         """ The program is initialized with -1, if it's something else we know its been filled already.
        The program is added in a spiral shape starting by moving to the right. Move down if left block 
        is filled and bottom is emtpy, or move left if top is filled, or move up if right is filled, else 
        move right."""
        
        if self.start:
            self.x += self.vocab_w
            self.start = False
        
        if self.x != 0 and self.program[self.x - 1, self.y] != -1 \
        and self.program[self.x, self.y + self.vocab_h] == -1:
            self.y += self.vocab_h
        elif self.y != 0 and self.program[self.x, self.y - 1] != -1:
            self.x -= self.vocab_w
        elif self.x + self.vocab_w != self.program_size and self.program[self.x + self.vocab_w, self.y] != -1:
            self.y -= self.vocab_h
        else:
            self.x += self.vocab_w