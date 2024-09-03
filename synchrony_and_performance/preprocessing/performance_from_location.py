import os
import sys
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import signal
import matplotlib.pyplot as plt


def read_data(path):
    return pd.read_pickle(os.path.join(path, 'epoched_raw_location.pkl'))

def get_performance_from_location(location_df):
    performance_df = copy.deepcopy(location_df)
    performance_df['performance'] = None

    ring_t = np.array(list(location_df.ringTime))
    start_t = np.array(list(location_df.startTime))

    min_time = np.sort(ring_t - start_t)[1]
    max_time = np.sort(ring_t - start_t)[-1]

    radius_dict = {'Easy': 4000,
                   'Medium': 3200,
                   'Hard': 2400}

    for i in range(len(location_df)):
        temp_location = location_df.iloc[i]
        ring_idx = np.where(temp_location.time == temp_location.ringTime)[0][0]
        x_diff = temp_location.ringY - temp_location.location[1,ring_idx]
        y_diff = temp_location.ringZ - temp_location.location[2,ring_idx]
        diviation = np.sqrt(x_diff**2 + y_diff**2) / radius_dict[temp_location.difficulty]
        relative_time = (temp_location.ringTime - temp_location.startTime - min_time) / (max_time - min_time)
        performance = 1 - relative_time * diviation
        performance_df.at[i, 'performance'] = performance
    return performance_df


if __name__ == '__main__':
    path = '../../data'

    pd.set_option('display.max_columns', None)
    location_df = read_data(path)
    performance_df = get_performance_from_location(location_df)

