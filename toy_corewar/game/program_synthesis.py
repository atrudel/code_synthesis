import numpy as np
import config

CFG = config.get_cfg()
CWCFG= CFG.settings.toy_corewar

class Instruction():
    def __init__(self, opcode, arg1, arg2, arg3):
        self.opcode = opcode
        self.arg1 = arg1
        self.arg2 = arg2
        self.arg3 = arg3
        
    def embeddings(self):
        # instruction embedding: one hot encoding
        instr_embedding = np.zeros(CWCFG.N_INSTRUCTIONS)
        instr_embedding[self.opcode - 1] = 1
        
        # variable embeddings: one hot encoding
        var_embedding = np.zeros((CWCFG.N_VARS, CWCFG.NUM_REGISTERS))
        # value encoding: direct scalar value
        val_embedding = np.zeros(1)
        
        # handle the special case of ld, which uses the value embedding
        if self.opcode == 1:
            var_embedding[1][self.arg2-1] = 1
            val_embedding[0] = self.arg1
        else:
            var_embedding[0][self.arg1-1] = 1
            var_embedding[1][self.arg2-1] = 1
            if self.arg3 is not None:
                var_embedding[2][self.arg3-1] = 1
        return np.concatenate((instr_embedding, var_embedding.reshape(-1), val_embedding))
        
    def to_byte_sequence(self):
        arg3 = self.arg3 if self.arg3 is not None else 0
        return np.array([self.opcode, self.arg1, self.arg2, arg3], dtype=int)

    def to_tuple(self):
        return self.opcode, self.arg1, self.arg2, self.arg3

    def padding_embedding():
        return np.zeros(CWCFG.N_INSTRUCTIONS + CWCFG.N_VARS * CWCFG.NUM_REGISTERS + 1)
    
    def print(self, file=None, end=None):
        labels = ['ld', 'st', 'add', 'sub']
        print(labels[self.opcode-1].ljust(3), file=file, end=' ')
        print("{} {} {}".format(str(self.arg1).ljust(2), str(self.arg2).ljust(2), str(self.arg3).ljust(4)),
             file=file, end=end)
        

class Program():
    def __init__(self, instructions=None, maxlen=CWCFG.MAX_LENGTH):
        self.instructions = instructions if instructions is not None else []
        self.maxlen = maxlen
    
    def add_instruction(self, instruction):
        '''Adds an instruction to the program and returns a value for the 'done' 
        parameter of the Environment (i.e. True if Program reached max length'''
        assert len(self.instructions) < self.maxlen
        self.instructions.append(instruction)
        return len(self.instructions) == self.maxlen
    
    def to_byte_sequence(self):
        # output np array corresponding to the sequence of bytes to load in ToyCorewar
        byte_sequences = [instr.to_byte_sequence() for instr in self.instructions]
        byte_sequences += [np.zeros(4, dtype=int)] * (self.maxlen - len(self.instructions))
        return np.concatenate(byte_sequences)
        
    def to_embedding_sequence(self):
        embeddings = [instr.embeddings() for instr in self.instructions]
        embeddings += [Instruction.padding_embedding()] * (self.maxlen - len(self.instructions))
        return embeddings
    
    def print(self, file=None, end=None):
        for instr in self.instructions:
            instr.print(file, end)
    
    def __getitem__(self, index):
        return self.instructions[index]

    def __len__(self):
        return len(self.instructions)
    
    def __iter__(self):
        for i in range(len(self.instructions)):
            yield Program(self.instructions[i])
