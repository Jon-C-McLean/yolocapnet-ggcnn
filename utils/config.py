import os

def parse_config(file):
    file = open(file, 'r')
    lines = file.readlines()
    lines = [line for line in lines if len(line) > 0 and line[0] != '#']
    lines = [line.rstrip().lstrip() for line in lines] # Remove whitespace
    
    blocks = []
    curr_block = {}

    for line in lines:
        if line[0] == '[':
            if len(curr_block) != 0:
                blocks.append(block)
                block = {}
            else:
                k,v = line.split('=')
                block[k.rstrip().lstrip()] = v.rstrip().lstrip()
            
    blocks.append(block)

    return blocks

