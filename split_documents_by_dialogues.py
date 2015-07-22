"""
Takes as input a dialogue file and splits it up by end-of-dialogue tokens and shuffles it </d>.

Example run:

    python split_documents_by_dialogues.py Data/Training_Shuffled_Dataset.txt Data/Training_SplitByDialogues_Dataset.txt

@author Iulian Vlad Serban
"""


from random import shuffle
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, help="Dialogue file; with one document (e.g. movie) per line")

parser.add_argument("output", type=str, help="Dialogue file; shuffled with one dialogue per line")
args = parser.parse_args()

if not os.path.isfile(args.input):
    raise Exception("Input file not found!")


data = open(args.input, 'r').readlines()

new_data = []
for l in data:
    s = l.split(' </d> </s> ')
    for i in range(len(s)-1):
        new_data.append(s[i] + ' </d> </s>')
        
shuffle(new_data)

f = open(args.output,'w')
for i in range(len(new_data)):
    f.write(new_data[i] + '\n')

f.close()
