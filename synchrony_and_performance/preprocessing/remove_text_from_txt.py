import re
import os
import sys
import copy
import numpy as np



def process_file(input_path, output_path):
    with open(input_path, 'r') as file:
        lines = file.readlines()
    
    processed_lines = []
    
    for line in lines:
        # Remove Rotation part
        line = re.sub(r'Rotation: P=[^ ]+ Y=[^ ]+ R=[^ ]+', '', line)
        # Extract only the numbers and keep the first 6 sets
        numbers = re.findall(r'-?\d+\.?\d*', line)[:6]
        if numbers:
            processed_lines.append(" ".join(numbers))
    
    with open(output_path, 'w') as file:
        file.write("\n".join(processed_lines))



if __name__ == '__main__':
    path = '../../data/raw_xdf/T16_S2/'
    for fileid in range(27,51):
        process_file((path + f"RingPositions_{fileid}.txt"), (path + f"RingPositions_{fileid}.txt"))
