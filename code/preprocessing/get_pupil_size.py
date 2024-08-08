import os
import sys
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import signal
import matplotlib.pyplot as plt


def read_data(path):
    pupil_df = pd.read_pickle(os.path.join(path, 'raw_pupilsize.pkl'))
    openness_df = pd.read_pickle(os.path.join(path, 'raw_openness.pkl'))
    location_df = pd.read_pickle(os.path.join(path, 'trialed_location.pkl'))
    epoched_location = pd.read_pickle(os.path.join(path, 'epoched_raw_location.pkl'))
    return pupil_df, openness_df, location_df, epoched_location

def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def extract_first_of_groups(numbers, threshold=30):
    # Initialize the result list with the first element
    result = [numbers[0]]

    # Iterate through the list, starting from the second element
    for i in range(1, len(numbers)):
        # Check if the current number is significantly larger than the previous one
        if numbers[i] - numbers[i - 1] > threshold:
            result.append(numbers[i])

    return result


def trial_raw_pupil(pupilsize_df, openness_df, location_df):

    trialed_raw_pupil = copy.deepcopy(location_df)
    trialed_raw_pupil = trialed_raw_pupil.drop(columns={'yawLocation','pitchLocation','thrustLocation'})
    trialed_raw_pupil['yawPupil'] = None
    trialed_raw_pupil['pitchPupil'] = None
    trialed_raw_pupil['thrustPupil'] = None
    trialed_raw_pupil['yawOpenness'] = None
    trialed_raw_pupil['pitchOpenness'] = None
    trialed_raw_pupil['thrustOpenness'] = None

    for i in tqdm(range(len(trialed_raw_pupil))):
        temp_location = trialed_raw_pupil.iloc[i]
        temp_pp = pupilsize_df.loc[(pupilsize_df.teamID == temp_location.teamID)
                        &(pupilsize_df.sessionID == temp_location.sessionID)]
        temp_op = openness_df.loc[(openness_df.teamID == temp_location.teamID)
                        &(openness_df.sessionID == temp_location.sessionID)]
        for role in ['Yaw', 'Pitch', 'Thrust']:

            temp_pp_time = temp_pp.loc[temp_pp.role == role].time.iloc[0]
            correct_pp_time = temp_pp_time - temp_pp_time[0]
            try:
                temp_trial_start_time, temp_trial_end_time = temp_location[f'{role.lower()}Time'][0], temp_location[f'{role.lower()}Time'][-1]
            except:
                print(f"No pupil data abliable for {role} in {temp_location.teamID} {temp_location.sessionID}")
                continue   
            if temp_location.teamID == 'T5' and  temp_location.sessionID == 'S2' and role == 'Thrust':  
                print(f'time error, use time of other roles instead')
                temp_trial_start_time, temp_trial_end_time = temp_location[f'yawTime'][0], temp_location[f'yawTime'][-1]
                

            trial_start = np.where(correct_pp_time <= temp_trial_start_time)[0][-1]
            trial_end = np.where(correct_pp_time >= temp_trial_end_time)[0][0]
            trialed_raw_pupil.at[i,f'{role.lower()}Pupil'] = temp_pp.loc[temp_pp.role == role].data.iloc[0].T[:,trial_start+1:trial_end]
            trialed_raw_pupil.at[i,f'{role.lower()}Openness'] = temp_op.loc[temp_op.role == role].data.iloc[0].T[:,trial_start+1:trial_end]
    return trialed_raw_pupil

def find_pupil_invalid_idxs(pupil_size, openness):
    invalid_l = np.array([]).astype(int)
    invalid_r = np.array([]).astype(int)
    # remove if pupil size is -1.0
    invalid_l = np.append(invalid_l, np.where(pupil_size[0,:] == 0)[0])
    invalid_r = np.append(invalid_r, np.where(pupil_size[1,:] == 0)[0])
    # remove dilation speed outliers and edge artications
    diff_size = np.abs(np.diff(pupil_size, axis=1))
    invalid_l = np.append(invalid_l,np.where(diff_size[0,:]>0.5)[0][:])
    invalid_r = np.append(invalid_r,np.where(diff_size[1,:]>0.5)[0][:])
    # remove blink: remove if eye openness is less than mean-std
    threshold = 0.3
    invalid_l = np.append(invalid_l, np.where(openness[0,:] < threshold)[0])
    invalid_r = np.append(invalid_r, np.where(openness[1,:] < threshold)[0])

    if len(invalid_l) != 0 or len(invalid_r) != 0:
        invalid_l = np.unique(invalid_l)
        invalid_r = np.unique(invalid_r)
    return invalid_l, invalid_r

def remove_inv(data, data_time, invalid_l, invalid_r):
    # compute number removed
    num_remove = round(get_sample_rate(data_time) * 5 / 100)
    invalid_lr = [invalid_l, invalid_r]
    for lr in range(2):
        if len(invalid_lr[lr]) != 0:
            for i in range(len(invalid_lr[lr])):
                # remove edge values first
                if invalid_lr[lr][i] < num_remove:
                    data[lr,0:num_remove] = np.nan
                elif invalid_lr[lr][i] > data_time.shape[0]-num_remove:
                    data[lr,-num_remove:] = np.nan
                else:
                    data[lr,invalid_lr[lr][i]] = np.nan
    return data

def linear_interp(data, start, end):
    return np.linspace(data[start], data[end], end-start)

def fill_missing(preprocessed_pupil_size, sample_rate):
    # fill the missing pupil data
    fill_miss_ps = copy.deepcopy(preprocessed_pupil_size)
    for lr in range(2):
        continuous_blocks = consecutive(np.where(np.isnan(preprocessed_pupil_size[lr,:]))[0])
        if len(continuous_blocks[0]) != 0: # have at least one block of missing data
            for i in range(len(continuous_blocks)): # loop through each missing data bolck
                # do not fill miss if missing is longer than 10 seconds
                if continuous_blocks[i].size > sample_rate * 10: 
                    # print('missing more than 10s pupil data, make pupil size equal to 0')
                    fill_miss_ps[lr,continuous_blocks[i][0]:continuous_blocks[i][-1]] = 0
                elif continuous_blocks[i][0] > 0 and continuous_blocks[i][-1] < preprocessed_pupil_size.shape[1]-1:
                    fill_start = continuous_blocks[i][0]-1
                    fill_end = continuous_blocks[i][-1]+1
                    result = linear_interp(preprocessed_pupil_size[lr,:], fill_start, fill_end)
                    fill_miss_ps[lr,fill_start:fill_end] = result
                elif continuous_blocks[i][0] > 0: 
                    # the last elemnt is out of range
                    fill_start = continuous_blocks[i][0]-1
                    fill_miss_ps[lr,fill_start+1:] = [fill_miss_ps[lr,fill_start]]*continuous_blocks[i].size
                elif continuous_blocks[i][-1] < preprocessed_pupil_size.shape[1]-1:
                    # the first elemnt is out of range
                    fill_end = continuous_blocks[i][-1]+1
                    fill_miss_ps[lr,:fill_end] = [fill_miss_ps[lr,fill_end]]*continuous_blocks[i].size
    return fill_miss_ps

def butter_lowpass_filter(data, cutoff, fs, order=2):
    # Design the Butterworth filter
    nyquist_freq = 0.5 * fs
    normal_cutoff = cutoff / nyquist_freq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    # Apply the filter to the pupil data
    filtered_pupil_data = signal.filtfilt(b, a, data)
    return filtered_pupil_data

def get_sample_rate(data_time):
    sample_rate = int(len(data_time) / (data_time[-1] - data_time[0]))
    if sample_rate > 0:
        # great, no gap in time
        return sample_rate
    else:
        gap_in_time = np.where(np.abs(np.diff(data_time)) > 0.03)[0]
        end_current_gap_idx = gap_in_time[1:]
        gap_i = gap_in_time[0]
        data_time[gap_i+1:] = data_time[gap_i] + (data_time[gap_i] - data_time[gap_i-1])
        return int(len(data_time) / (data_time[-1] - data_time[0]))


def change_sample_rate(data, time, target_sample_rate, resample_axis):
    trial_time_length = time[-1]-time[0]
    resampled_data = signal.resample(data, round(target_sample_rate * trial_time_length), time, axis=resample_axis)
    return resampled_data

def butter_lowpass_filter(data, cutoff, fs, order=2):
    # Design the Butterworth filter
    nyquist_freq = 0.5 * fs
    normal_cutoff = cutoff / nyquist_freq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    # Apply the filter to the pupil data
    filtered_pupil_data = signal.filtfilt(b, a, data)
    return filtered_pupil_data

def process_trialed_pupil(trialed_pupil_df):
    order = 4
    cutoff = 4
    preprocess_pupil_df = copy.deepcopy(trialed_pupil_df)
    preprocess_pupil_df = preprocess_pupil_df.drop(columns={'yawTime',
       'pitchTime', 'thrustTime', 'yawPupil', 'pitchPupil', 'thrustPupil',
       'yawOpenness', 'pitchOpenness', 'thrustOpenness'})
    preprocess_pupil_df['yawProcessedPupil'] = None
    preprocess_pupil_df['pitchProcessedPupil'] = None
    preprocess_pupil_df['thrustProcessedPupil'] = None
    preprocess_pupil_df['yawTime'] = None
    preprocess_pupil_df['pitchTime'] = None
    preprocess_pupil_df['thrustTime'] = None
    for role in ['Yaw', 'Pitch', 'Thrust']:
        for i in tqdm(range(len(trialed_pupil_df))):
            temp_pp = trialed_pupil_df.iloc[i][f'{role.lower()}Pupil']
            temp_op = trialed_pupil_df.iloc[i][f'{role.lower()}Openness']
            temp_time = trialed_pupil_df.iloc[i][f'{role.lower()}Time']
            if temp_pp is None:
                continue
            invalid_l, invalid_r = find_pupil_invalid_idxs(temp_pp, temp_op)
            preprocessed_pupil_size = remove_inv(temp_pp, temp_time, invalid_l, invalid_r)
            fill_missed_pupil_size = fill_missing(preprocessed_pupil_size, get_sample_rate(temp_time))
            averaged_pupil_size = np.nanmean(fill_missed_pupil_size, axis=0)
            resampled_pupil_size, resampled_time = change_sample_rate(averaged_pupil_size, temp_time, 60, 0)
            filtered_pupil_size = butter_lowpass_filter(resampled_pupil_size, cutoff, 60, order)
            z_scored_pupil_size = (filtered_pupil_size - np.mean(filtered_pupil_size)) / np.std(filtered_pupil_size)
            preprocess_pupil_df.at[i,f'{role.lower()}ProcessedPupil'] = z_scored_pupil_size
            preprocess_pupil_df.at[i,f'{role.lower()}Time'] = resampled_time
   
    return preprocess_pupil_df

def epoch_pupil(preprocess_pupil_df, epoched_location):
    epoched_pupil_df = copy.deepcopy(epoched_location)
    epoched_pupil_df = epoched_pupil_df.drop(columns={'ringX', 'ringY', 'ringZ', 'location', 'time', 'startTime'})
    epoched_pupil_df['yawPupil'] = None
    epoched_pupil_df['pitchPupil'] = None
    epoched_pupil_df['thrustPupil'] = None
    for i in tqdm(range(len(epoched_pupil_df))):
        temp_epoch_df = epoched_pupil_df.iloc[i]
        processed_pp = preprocess_pupil_df.loc[(preprocess_pupil_df.teamID == temp_epoch_df.teamID)
                              & (preprocess_pupil_df.sessionID == temp_epoch_df.sessionID)
                              & (preprocess_pupil_df.trialID == temp_epoch_df.trialID)]
        for role in ['yaw', 'pitch', 'thrust']:
            ring_idx = np.where(processed_pp[f"{role}Time"].iloc[0] <= temp_epoch_df.ringTime)[0]
            if len(ring_idx) > 0:
                ring_idx = ring_idx[-1]
            else:
                continue
            temp_epoched_pp = processed_pp[f"{role}ProcessedPupil"].iloc[0][ring_idx-90:ring_idx]
            if ring_idx >= 90 and ~np.isnan(temp_epoched_pp).all():
                epoched_pupil_df.at[i, f"{role}Pupil"] = temp_epoched_pp

    return epoched_pupil_df.dropna().reset_index(drop=True)

if __name__ == '__main__':
    path = '../../data'
    pd.set_option('display.max_columns', None)
    pupil_df, openness_df, location_df, epoched_location = read_data(path)
    trialed_pupil_df = trial_raw_pupil(pupil_df, openness_df, location_df)
    preprocess_pupil_df = process_trialed_pupil(trialed_pupil_df)
    epoched_pupil_df = epoch_pupil(preprocess_pupil_df, epoched_location)
    epoched_pupil_df.to_pickle(os.path.join(path, 'epoched_pupil.pkl'))




    # action_df.to_pickle(os.path.join(path, 'epoched_action.pkl'))
    