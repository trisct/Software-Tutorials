import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str)
args = parser.parse_args()



header = ''
in_header = True
v_list = []
f_list = []

with open(args.input, 'r') as plyfile:
    while True:
        line = plyfile.readline()
        if len(line) == 0:
            break
        
        if in_header:
            header += line
        else:
            nums = line[:-1].split(' ')
        




        print(line)