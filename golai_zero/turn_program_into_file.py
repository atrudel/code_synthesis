def turn_program_into_file(program, filename="program.rle"):
    
    """ Saves a player program, a numpy array, into the RLE text format. 
    Here are more details about the format: http://www.conwaylife.com/wiki/RLE
    """
    name= ""
    author= ""
    comments = ""
    player_width = program.shape[0]
    player_height = program.shape[1]
      
    file = "#N " + name + "\n"
    file += "#C " + comments + "\n"
    file += "#O " + author + "\n"
    file += "x = " + str(player_width) + ", y= " + str(player_height) + "\n"
    
    linebreak = 0
    for i in range(player_height):
        counter = 1
        for k in range(player_width - 1):
            if program[i][k] == 0:
                data = "b"
            else:
                data = "o"
            
            if program[i][k + 1] == 0:
                data_2 = "b"
            else:
                data_2 = "o"
            
            if data != data_2 or k == (player_width - 2):
                if data == data_2:
                    counter += 1
                
                if counter > 1:
                    file += (str(counter))
                    linebreak += len(str(counter)) - 1
                
                file += data
                if k == (player_width - 2) and data != data_2:
                    file += data_2
                
                linebreak += 1
                if linebreak > 70:
                    file += "\n"
                    linebreak = 0
                
                counter = 0
                
            counter += 1
        linebreak += 1
        if i != player_height - 1:
            file += "$"     
    file += "!"
    f = open(filename, 'w') 
    f.write(str(file))
    f.close() 