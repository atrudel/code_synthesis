def turn_program_into_file(program, player_size, experiment_name, author, comments, dir_name):
    
    file = "#N " + experiment_name + "\n"
    file += "#C " + comments + "\n"
    file += "#O " + author + "\n"
    file += "x = " + str(player_size) + ", y= " + str(player_size) + "\n"
    
    linebreak = 0
    for i in range(player_size):
        counter = 1
        for k in range(player_size - 1):
            if program[i][k] == 0:
                data = "b"
            else:
                data = "o"
            
            if program[i][k + 1] == 0:
                data_2 = "b"
            else:
                data_2 = "o"
            
            if data != data_2 or k == (player_size - 2):
                if data == data_2:
                    counter += 1
                
                if counter > 1:
                    file += (str(counter))
                    linebreak += len(str(counter)) - 1
                
                file += data
                if k == (player_size - 2) and data != data_2:
                    file += data_2
                
                linebreak += 1
                if linebreak > 70:
                    file += "\n"
                    linebreak = 0
                
                counter = 0
                
            counter += 1
        file += "$"
        linebreak += 1
    file += "!"
    print(file)
    f = open(experiment_name + ".rle", 'w') 
    f.write(str(file))
    f.close() 