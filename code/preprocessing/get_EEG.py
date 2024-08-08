import os
import mne
import sys
import json
import copy
import pickle
import pyprep
import numpy as np
import pandas as pd
import heartpy as hp
from tqdm import tqdm
from scipy import signal
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
from mne_icalabel import label_components
from pyprep import NoisyChannels

def bandpass_filter(eeg_data, low_freq, high_freq, sampling_rate):
    nyquist_freq = 0.5 * sampling_rate
    low = low_freq / nyquist_freq
    high = high_freq / nyquist_freq
    b, a = signal.butter(4, [low, high], btype='band')
    filtered_data = signal.filtfilt(b, a, eeg_data, axis=0)
    return filtered_data
def ekg_notch_filter(data):
    b, a = signal.iirnotch(60, 80, 256)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data
def highpass_filter(ekg_data, cutoff_freq, sampling_rate):
    """
    Apply a low-pass filter to EKG data.
    
    Args:
        ekg_data (numpy.ndarray): EKG data.
        cutoff_freq (float): Cutoff frequency for the low-pass filter in Hz (commonly around 50 Hz).
        sampling_rate (int): Sampling rate of the EKG data (samples per second).
    
    Returns:
        numpy.ndarray: Filtered EKG data.
    """
    # Design a Butterworth low-pass filter
    nyquist_freq = 0.5 * sampling_rate
    b, a = signal.butter(1, cutoff_freq / nyquist_freq, btype='low', analog=False)
    
    # Apply the filter to the EKG data
    filtered_data = signal.hfilter(b, a, ekg_data)
    return filtered_data

def run_preprocessing(eeg_data, n_components, channel_locs, file_name):
    sampling_rate = 256  

    # Create an MNE Raw object from the EEG data
    info = mne.create_info(ch_names=channel_locs.ch_names, sfreq=sampling_rate, ch_types='eeg', verbose=False)
    raw = mne.io.RawArray(np.array(eeg_data.T/1e6, dtype="float64"), info, verbose=False)
    montage = mne.channels.make_dig_montage(ch_pos=channel_locs.get_positions()['ch_pos'], coord_frame='head')
    raw.set_montage(montage, verbose=False)

    # bad channel removal
    all_noisy_channel_lst = []
    noisy_eeg = NoisyChannels(raw, random_state=1)
    noisy_eeg.find_bad_by_correlation()
    all_noisy_channel_lst += noisy_eeg.bad_by_correlation
    noisy_eeg.find_bad_by_deviation()
    all_noisy_channel_lst += noisy_eeg.bad_by_deviation
    noisy_eeg.find_bad_by_ransac()
    all_noisy_channel_lst += noisy_eeg.bad_by_ransac
    all_noisy_channel_lst = list(np.unique(all_noisy_channel_lst))
    print("################################################################################")
    print("Bad channels original: {}".format(all_noisy_channel_lst))
    print("################################################################################")
    # remove and interpolate bad channel
    raw.info['bads'] = all_noisy_channel_lst
    raw = raw.copy().interpolate_bads(reset_bads=True, verbose=False)

    # filter data
    raw = raw.filter(0.5, 50, verbose=False)

    # Re-referencing
    raw = raw.set_eeg_reference([])

    # Initialize ICA
    ica = ICA(n_components=None, random_state=1, max_iter=800, method='infomax', fit_params=dict(extended=True))
    # Fit ICA to the raw EEG data
    ica.fit(raw, verbose=False)
    

    IC_labels = label_components(raw, ica, method='iclabel')


    # Threshold for the "brain" label probability
    reject_threshold = 0.8
    brain_reject_threshold = 0.2

    # Initialize a list to store the indices of ICs to be excluded
    exclude_ic_indices = []

    # Iterate through the labels_info and exclude ICs that don't meet the threshold
    for idx, (label, proba) in enumerate(zip(IC_labels['labels'], IC_labels['y_pred_proba'])):
        if label != 'brain' and proba > reject_threshold:
            exclude_ic_indices.append(idx)
        elif label == 'brain' and proba < brain_reject_threshold:
            exclude_ic_indices.append(idx)

    exclude_ic_indices = np.unique(exclude_ic_indices)

    # Exclude the identified ICs from further analysis
    ica.exclude = exclude_ic_indices
    
    # Apply ICA to the EEG data to remove artifact components
    cleaned_eeg_data = ica.apply(raw)
    clean_fig = cleaned_eeg_data.plot(show=False, verbose=False)
    pickle.dump(clean_fig, open(f'{path}/eeg_plots/eeg_data_clean_plot_{file_name}.pickle', 'wb')) 
    plt.close()
    return cleaned_eeg_data.get_data()

def process_eeg(eeg_data_df, channel_locs, path):
    preprocessed_eeg_lst = []
    for i in tqdm(range(48,len(eeg_data_df))):#
        eeg_data = eeg_data_df.iloc[i]['data'][:,6:26]
        cleaned_eeg_data = run_preprocessing(eeg_data, 20, channel_locs, i)
        preprocessed_eeg_lst.append(cleaned_eeg_data)
        temp_df = eeg_data_df.iloc[i].drop(columns=['data'])
        temp_df['processed_eeg'] = cleaned_eeg_data
        temp_df.to_pickle(f"temp_files/{i}.pkl")

    preprocessed_eeg_df = eeg_data_df.drop(columns=['data'])
    preprocessed_eeg_df['processed_eeg'] = preprocessed_eeg_lst

    return preprocessed_eeg_df

def epoch_eeg(preprocessed_eeg_df, epoched_location):
    epoched_eeg = []
    invalid_idx_lst = []
    epoched_location = epoched_location[epoched_location.role != 'Thrust'].reset_index(drop=True)
    for i in tqdm(range(len(epoched_location))):
        if len(epoched_location.location.iloc[i][0])< 10:
            invalid_idx_lst.append(i)
            continue
        query_key = epoched_location.iloc[i][['teamID', 'sessionID', 'role']]
        query_string = f"teamID == '{query_key[0]}' and sessionID == '{query_key[1]}' and role == '{query_key[2]}'"
        try:
            temp_eeg = preprocessed_eeg_df.query(query_string).processed_eeg.iloc[0]
            temp_eeg_time = preprocessed_eeg_df.query(query_string).time.iloc[0]
        except:
            invalid_idx_lst.append(i)
            continue
        # time start from zero
        if 'ai' not in path:
            temp_eeg_time = temp_eeg_time - temp_eeg_time[0]
        try:
            ring_idx = np.where(temp_eeg_time >= epoched_location.ring_time.iloc[i])[0][0]
        except:
            invalid_idx_lst.append(i)
            continue
        if ring_idx - 512 > 0 and ring_idx + 513 < len(temp_eeg_time)-1:
            epoched_eeg.append(temp_eeg[:, ring_idx-512:ring_idx+513])
        else:
            invalid_idx_lst.append(i)
    epoched_eeg_df = epoched_location.drop(columns=['location', 'ring_time', 'time'])
    epoched_eeg_df = epoched_eeg_df.drop(invalid_idx_lst).reset_index(drop=True)
    epoched_eeg_df['eeg'] = epoched_eeg
    return epoched_eeg_df


if __name__ == '__main__':
    path = '../../data/' 
    pd.set_option('display.max_columns', None)

    channel_locs = mne.channels.read_custom_montage('chan_locs.sfp')
    eeg_df = pd.read_pickle(os.path.join(path, 'raw_eeg.pkl'))
    eeg_df = eeg_df.reset_index(drop=True)

    preprocessed_eeg_df = process_eeg(eeg_df, channel_locs, path)
    preprocessed_eeg_df.to_pickle(os.path.join(path, 'preprocessed_eeg.pkl' ))
    preprocessed_eeg_df = pd.read_pickle(os.path.join(path, 'preprocessed_eeg.pkl' ))

    epoched_location = pd.read_pickle(os.path.join(path, 'epoched_data', 'epoched_location.pkl'))     
    epoched_eeg_df = epoch_eeg(preprocessed_eeg_df, epoched_location)
    epoched_eeg_df.to_pickle(os.path.join(path, 'epoched_data', 'epoched_eeg.pkl' ))




