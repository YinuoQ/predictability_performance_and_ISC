import os
import sys
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split




def main():
    yaw_out = np.load('yaw/split_0/test/train_output.npy')
    yaw_in = np.load('yaw/split_0/train/train_output.npy')
    import IPython
    IPython.embed()
    assert False
if __name__ == '__main__':
    main()