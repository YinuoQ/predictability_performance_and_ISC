import os
import mne
import sys
import json
import copy
import glob
import wave
import librosa
import webrtcvad
import numpy as np
import pandas as pd
from tqdm import tqdm
import noisereduce as nr
from scipy import signal
import matplotlib.pyplot as plt


def simple_vad(audio, threshold=0.01, close_gap=0.001, sample_rate=11025):
    """
    Simple voice activity detection with additional feature to merge close speech segments.

    :param audio: Audio signal array
    :param threshold: Amplitude threshold for speech detection
    :param close_gap: Time gap in seconds to merge close speech segments
    :param sample_rate: Sample rate of the audio
    :return: Speech detection array
    """
    # Compute the absolute amplitude and detect initial speech frames
    amplitude = audio
    speech_frames = np.where(amplitude > threshold, 1, 0)

    # Number of samples corresponding to the close_gap
    close_gap_samples = int(close_gap * sample_rate)

    # Iterate through detected speech frames and merge close segments
    i = 0
    while i < len(speech_frames):
        if speech_frames[i] == 1:
            j = i + 1
            # Find the end of the current speech segment
            while j < len(speech_frames) and speech_frames[j] == 1:
                j += 1
            # Look ahead to see if the next speech segment is close
            next_speech = j
            while next_speech < len(speech_frames) and next_speech - j <= close_gap_samples:
                if speech_frames[next_speech] == 1:
                    # If close enough, mark all in-between frames as speech
                    speech_frames[i:next_speech+1] = 1
                    break
                next_speech += 1
            # Move to the next segment
            i = next_speech
        else:
            i += 1

    return speech_frames

def filter_short_speech(speech_detection, time_data, min_duration):
    # Convert min_duration from seconds to samples
    min_samples = min_duration * 11025

    # Initialize a filtered speech detection array
    filtered_speech = np.copy(speech_detection)

    # Iterate through the speech detection array and filter out short speech segments
    current_segment_length = 0
    for i in range(len(speech_detection)):
        if speech_detection[i] == 1:
            current_segment_length += 1
        else:
            if current_segment_length > 0 and current_segment_length < min_samples:
                # If the segment is shorter than the minimum duration, set it to no speech
                filtered_speech[i-current_segment_length:i] = 0
            current_segment_length = 0

    return filtered_speech

def change_sample_rate(data, time, target_sample_rate, resample_axis):
    trial_time_length = time[-1]-time[0]
    resampled_data = signal.resample(data, round(target_sample_rate * trial_time_length), time, axis=resample_axis)
    return resampled_data


def down_sample_speech(data, speech_time):
    if data is None:
        return None
    else:      
        resampled_data, resampled_time = change_sample_rate(data, speech_time, 60, 0)
        resampled_data[resampled_data < 0.5] = 0
        resampled_data[resampled_data != 0] = 1
        return resampled_data, resampled_time

def process_speech(speech_data_df, path):
    processed_speech_lst = []
    for i in tqdm(range(len(speech_data_df))):
        temp_audio = np.squeeze(speech_data_df['data'].iloc[i])
        data_rn = nr.reduce_noise(y=temp_audio, sr=11025,  y_noise = temp_audio[:5*11025])
        speech_time = speech_data_df['time'].iloc[i]
        speech_detection = simple_vad(data_rn**2, threshold=2*np.std(data_rn**2), close_gap=1)
        filtered_speech_detection = filter_short_speech(speech_detection, speech_time, 0.1)
        down_sampled_speech_event, down_sampled_time = down_sample_speech(filtered_speech_detection, speech_time)
        if np.sum(down_sampled_speech_event) < 10:
            # do not have valide audio
            speech_data_df.iat[i, 5] = None
            speech_data_df.iat[i, 6] = None
        else:
            speech_data_df.iat[i, 5] = down_sampled_speech_event
            speech_data_df.iat[i, 6] = down_sampled_time
    return speech_data_df

def get_start_end_idx_of_an_epoch(data_time, trial_time, start=True):
    if trial_time[1] - trial_time[0] > 2:
        trial_time[0] = trial_time[1] - (trial_time[2] - trial_time[1])
    if start: 
        if data_time[-1] < trial_time[0]:
            return None
        else:
            trial_start_idx = np.where(trial_time[0] >= data_time)[0][-1]
        return trial_start_idx
    else: #for getting end index
        if data_time[0] > trial_time[-1]:
            return None
        else:
            if np.sum(trial_time[-1]<= data_time)>0:
                trial_end_idx = np.where(trial_time[-1]<= data_time)[0][0]
            elif np.abs(data_time[-1] - trial_time[-1]) < 0.1:
                trial_end_idx = len(data_time)-1
            else:
                return None
        return trial_end_idx

def epoche_speech_event(speech_data_df, epoched_location):
    epoched_speech_event_df = copy.deepcopy(epoched_location)
    epoched_speech_event_df = epoched_speech_event_df.drop(columns=['location', 'ringTime', 'time', 'startTime', 'ringX', 'ringY', 'ringZ'])
    epoched_speech_event_df['yawSpeech'] = None
    epoched_speech_event_df['pitchSpeech'] = None
    epoched_speech_event_df['thrustSpeech'] = None

    for i in tqdm(range(len(epoched_speech_event_df))):
        for role in ['Yaw', 'Pitch', 'Thrust']:
            query_key = epoched_location.iloc[i][['teamID', 'sessionID']]
            query_string = f"teamID == '{query_key[0]}' and sessionID == '{query_key[1]}' and role == '{role}'"

            temp_speech = speech_data_df.query(query_string)
            if len(temp_speech) > 0:
                temp_speech_event = temp_speech.data.iloc[0]
                temp_speech_time = temp_speech.time.iloc[0]
            if temp_speech_time is None or temp_speech is None:
                continue
            else:
                temp_speech_time = temp_speech_time - temp_speech_time[0]
                ring_idx = np.where(epoched_location.iloc[i].ringTime <= temp_speech_time)[0]
                if len(ring_idx) > 0:
                    ring_idx = ring_idx[0]
                else:
                    continue
                if ring_idx >= 90:
                    epoched_speech_event_df.at[i,f'{role.lower()}Speech'] = temp_speech_event[ring_idx-90:ring_idx]
    
    return epoched_speech_event_df.dropna().reset_index(drop=True)


def main():
    path = '../../data'
    pd.set_option('display.max_columns', None)
    speech = pd.read_pickle(os.path.join(path, 'raw_audio.pkl'))
    speech_data_df = process_speech(speech, path)
    epoched_location = pd.read_pickle(os.path.join(path, 'epoched_raw_location.pkl'))  
    epoched_speech_event = epoche_speech_event(speech_data_df, epoched_location)  
    epoched_speech_event.to_pickle(os.path.join(path, 'epoched_speech_event.pkl'))

if __name__ == '__main__':
    main()



