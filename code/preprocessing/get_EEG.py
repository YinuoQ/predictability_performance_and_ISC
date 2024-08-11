import os
import mne
import sys
import json
import copy
import pickle
import pyprep
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import signal
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
from mne_icalabel import label_components
from pyprep import NoisyChannels


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
    fig,ax = plt.subplots()
    cleaned_eeg_data.plot(show=False, verbose=False)
    pickle.dump(fig, open(f'{path}/eeg_plots/eeg_data_clean_plot_{file_name}.fig.pickle', 'wb')) 
    plt.close()
    return cleaned_eeg_data.get_data()

def process_eeg(eeg_data_df, channel_locs, path):
    preprocessed_eeg_lst = []
    for i in tqdm(len(99)):#
        eeg_data = eeg_data_df.iloc[i]['data'][:,6:26]
        cleaned_eeg_data = run_preprocessing(eeg_data, 20, channel_locs, i)
        preprocessed_eeg_lst.append(cleaned_eeg_data)
        temp_df = eeg_data_df.iloc[i].drop(columns=['data'])
        temp_df['processed_eeg'] = cleaned_eeg_data
        temp_df.to_pickle(f'temp_files/{i}.pkl')
    preprocessed_eeg_df = eeg_data_df.drop(columns=['data'])
    preprocessed_eeg_df['processed_eeg'] = preprocessed_eeg_lst

    return preprocessed_eeg_df

def epoch_eeg(preprocessed_eeg_df, epoched_location):

    epoched_eeg = copy.deepcopy(epoched_location)
    epoched_eeg = epoched_eeg.drop(columns=['location', 'ringTime', 'time', 'startTime', 'ringX', 'ringY', 'ringZ'])
    epoched_eeg['yawEEG'] = None
    epoched_eeg['pitchEEG'] = None
    epoched_eeg['thrustEEG'] = None
    
    for i in tqdm(range(len(epoched_eeg))):
        for role in ['Yaw', 'Pitch', 'Thrust']:
            query_key = epoched_location.iloc[i][['teamID', 'sessionID']]
            query_string = f"teamID == '{query_key[0]}' and sessionID == '{query_key[1]}' and role == '{role}'"

            temp_eeg = preprocessed_eeg_df.query(query_string)
            if len(temp_eeg) > 0:
                temp_eeg_event = temp_eeg.processed_eeg.iloc[0]
                temp_eeg_time = temp_eeg.time.iloc[0]
            else:
                continue
            if temp_eeg_time is None or temp_eeg is None:
                continue
            else:
                temp_eeg_time = temp_eeg_time - temp_eeg_time[0]
                ring_idx = np.where(epoched_location.iloc[i].ringTime <= temp_eeg_time)[0]
                if len(ring_idx) > 0:
                    ring_idx = ring_idx[0]
                else:
                    continue
                if ring_idx >= 384:
                    epoched_eeg.at[i,f'{role.lower()}EEG'] = temp_eeg_event[:,ring_idx-384:ring_idx]     
    return epoched_eeg.dropna().reset_index(drop=True)

if __name__ == '__main__':
    path = '../../data/' 
    pd.set_option('display.max_columns', None)

    # channel_locs = mne.channels.read_custom_montage('chan_locs.sfp')
    # eeg_df = pd.read_pickle(os.path.join(path, 'raw_eeg.pkl'))
    # eeg_df = eeg_df.reset_index(drop=True)

    # preprocessed_eeg_df = process_eeg(eeg_df, channel_locs, path)

    # preprocessed_eeg_df = pd.DataFrame()
    # for i in tqdm(range(97)):      
    #     temp_pickle = pd.read_pickle(f'temp_files/{i}.pkl')
    #     preprocessed_eeg_df = pd.concat([preprocessed_eeg_df, temp_pickle], axis=1, ignore_index=True)
    # preprocessed_eeg_df.T.to_pickle(os.path.join(path, 'preprocessed_eeg.pkl' ))
    
    # import IPython
    # IPython.embed()
    # assert False
    preprocessed_eeg_df = pd.read_pickle(os.path.join(path, 'preprocessed_eeg.pkl' ))
    epoched_location = pd.read_pickle(os.path.join(path, 'epoched_raw_location.pkl'))     
    epoched_eeg_df = epoch_eeg(preprocessed_eeg_df, epoched_location)
    epoched_eeg_df.to_pickle(os.path.join(path, 'epoched_eeg.pkl' ))




