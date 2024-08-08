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

def remove_noisy_points(data):
    clean_data = copy.deepcopy(data)
    # Handle edge cases for very small arrays
    if len(data) < 3:
        return data
    # Iterate through the array, starting from the second element and ending at the second-to-last
    for i in range(1, len(data) - 1):
        # If the current element is different from both its neighbors
        if data[i] != data[i - 1] and data[i] != data[i + 1]:
            # Replace the current element with the previous one (could also be the next one)
            clean_data[i] = clean_data[i - 1]
    clean_data[-1] = clean_data[-2]
    return clean_data

def down_sample_and_epoch_action(data, data_time, target_sample_rate, ring_time, resample_axis=0):
    trial_time_length = data_time[-1]-data_time[0]
    resampled_data, resampled_time = signal.resample(data, round(target_sample_rate * trial_time_length), data_time, axis=resample_axis, window=10)
    ring_idx = np.where(resampled_time <= ring_time)[0][-1]
    resampled_data[resampled_data >= 0.5] = 1
    resampled_data[resampled_data <= -0.5] = -1
    resampled_data[np.where(np.abs(resampled_data) != 1)] = 0
    if ring_idx >= 90:
        resampled_data = resampled_data[ring_idx-90:ring_idx]
    else:
        resampled_data = None
    return resampled_data

def get_action(location_df):
    performance_df = copy.deepcopy(location_df)
    performance_df['yawAction'] = None
    performance_df['pitchAction'] = None
    performance_df['thrustAction'] = None

    for i in tqdm(range(len(location_df))):
        temp_location = performance_df.iloc[i]
        locationX = temp_location.location[0]
        locationY = temp_location.location[1]
        locationZ = temp_location.location[2]

        thrust_action = np.zeros(len(locationX)-1)
        thrust_action[np.where((np.diff(locationX) == 200) | (np.diff(locationX) == 400))] = 1
        thrust_action[np.where((np.diff(locationX) == 80) | (np.diff(locationX) == 160))] = -1
        thrust_action = np.insert(thrust_action,-1,thrust_action[-1])
        thrust_action = down_sample_and_epoch_action(thrust_action, temp_location.time, 60, temp_location.ringTime)

        yaw_action = np.zeros(len(locationY)-1)
        y_diff = remove_noisy_points(np.diff(locationY))
        yaw_action[np.where(y_diff <= -34)] = -1
        yaw_action[np.where(y_diff >= 34)] = 1
        yaw_action = np.insert(yaw_action,-1,yaw_action[-1])
        yaw_action = down_sample_and_epoch_action(yaw_action, temp_location.time, 60, temp_location.ringTime)

        pitch_action = np.zeros(len(locationZ)-1)
        z_diff = remove_noisy_points(np.diff(locationZ))
        pitch_action[np.where(z_diff <= -34)] = -1
        pitch_action[np.where(z_diff >= 34)] = 1
        pitch_action = np.insert(pitch_action,-1,pitch_action[-1])
        pitch_action = down_sample_and_epoch_action(pitch_action, temp_location.time, 60, temp_location.ringTime)

        performance_df.at[i,'yawAction'] = yaw_action
        performance_df.at[i,'pitchAction'] = pitch_action
        performance_df.at[i,'thrustAction'] = thrust_action


    performance_df = performance_df.drop(columns={'communication', 'difficulty', 'ringX', 
                                                  'ringY', 'ringZ', 'location', 'time', 
                                                  'startTime', 'ringTime'})
    performance_df = performance_df.dropna().reset_index(drop=True)

    return performance_df
if __name__ == '__main__':
    path = '../../data'
    pd.set_option('display.max_columns', None)
    location_df = read_data(path)
    action_df = get_action(location_df)
    action_df.to_pickle(os.path.join(path, 'epoched_action.pkl'))
    